"""Shared regular queries with matched total decoder-layer budget."""
from typing import Any

import torch
from torch import Tensor, nn
from dual_model import DualQueryDecoupledModel as ReferenceModel
from dual_model import sinusoidal_position_encoding


class SharedQueryModel(ReferenceModel):
    """Use one final query tensor for masks and functions; preserve all supervision."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        layers = list(self.seg_decoder.layers) + list(self.func_decoder.layers)
        self.seg_decoder.layers = nn.ModuleList(layers)
        self.seg_decoder.num_layers = len(layers)
        del self.func_decoder
        del self.func_prototype_emb

    def forward(self, residue_feat: Tensor, residue_mask: Tensor,
                gt_spans: Any = None, gt_func_targets: Any = None,
                training: bool = False) -> dict[str, Any]:
        """Decode isolated regular/DN/GT groups; share the regular state across heads."""
        batch, length, width = residue_feat.shape
        q = self.num_queries
        pad_mask = residue_mask == 0
        memory = self.encoder(residue_feat, src_key_padding_mask=pad_mask)
        scores = self.query_score_head(memory).squeeze(-1)
        indices = scores.masked_fill(pad_mask, -1e4).topk(q, dim=1).indices
        indices = indices.sort(dim=1).values
        content = memory.gather(1, indices.unsqueeze(-1).expand(-1, -1, width))
        seq_lens = residue_mask.sum(dim=1, keepdim=True).float()
        anchors = self.anchor_mlp((indices.float() / seq_lens.clamp(min=1)).unsqueeze(-1))
        groups = [content + anchors]
        sizes = [q]
        dn_masks = None
        gt_labels = gt_valid = None
        if training and gt_spans is not None and self.num_dn_queries > 0:
            dn, dn_masks = self._build_dn_queries(memory, residue_mask, gt_spans, seq_lens)
            groups.append(dn)
            sizes.append(dn.size(1))
        if training and gt_func_targets is not None and self.num_gt_func_queries > 0:
            gt, gt_labels, gt_valid = self._build_gt_func_queries(memory, gt_func_targets)
            groups.append(gt)
            sizes.append(gt.size(1))
        queries = torch.cat(groups, dim=1)
        isolation = torch.ones(queries.size(1), queries.size(1),
                               dtype=torch.bool, device=queries.device)
        offset = 0
        for size in sizes:
            isolation[offset:offset + size, offset:offset + size] = False
            offset += size
        decoded = self.seg_decoder(queries, memory, tgt_mask=isolation,
                                   memory_key_padding_mask=pad_mask)
        z = decoded[:, :q]
        pe = sinusoidal_position_encoding(length, width, device=memory.device).unsqueeze(0)
        mask_feat = torch.sigmoid(self.mask_gate(memory)) * (residue_feat + self.pe_scale * pe)
        masks = torch.einsum('bqd,bld->bql', self.mask_embed(z), mask_feat)
        func = self.func_head(z)
        objectness = self.objectness_head(z).squeeze(-1)
        per_domain = self.fusion_mlp(torch.cat([z, z], dim=-1))
        protein = torch.einsum('bq,bqd->bd', objectness.softmax(dim=1), per_domain)
        output = dict(mask_logits=masks, func_logits=func, class_logits=func,
                      validity_logits=self.validity_head(z).squeeze(-1),
                      query_scores=scores, objectness_logits=objectness,
                      seg_queries=z, func_queries=z, per_domain_repr=per_domain,
                      protein_repr=protein,
                      protein_logits=func.max(dim=1).values + self.protein_head(protein))
        offset = q
        if dn_masks is not None:
            end = offset + self.num_dn_queries
            output['dn_mask_logits'] = torch.einsum(
                'bkd,bld->bkl', self.mask_embed(decoded[:, offset:end]), mask_feat)
            output['dn_gt_masks'] = dn_masks
            offset = end
        if gt_labels is not None:
            gt_state = decoded[:, offset:offset + self.num_gt_func_queries]
            output.update(gt_func_logits=self.func_head(gt_state), gt_func_labels=gt_labels,
                          gt_validity_logits=self.validity_head(gt_state).squeeze(-1),
                          gt_validity_labels=gt_valid)
        return output


# Existing Lightning/checkpoint evaluator imports this compatibility name.
DualQueryDecoupledModel = SharedQueryModel
__all__ = ['SharedQueryModel', 'DualQueryDecoupledModel']
