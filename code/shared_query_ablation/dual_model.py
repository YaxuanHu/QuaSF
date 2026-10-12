r"""
Dual-query decoupled region--function model (QuerySplat-inspired).

Motivation
----------
The previous FunDSQ model (see ``QuerySelectionMaskDinoV2``) uses a *single*
query per candidate domain, and that query drives two heads at once: a mask
head (domain segmentation / "where") and a classification head (domain function
/ "what"). This entangles two very different inductive biases in one
representation.

This module follows the central idea of QuerySplat (arXiv:2608.01186) --
*decoupling geometry and appearance into two query branches* -- and applies it
to proteins:

    * a **segmentation query stream** ("geometry branch") that is position
      anchored and learns the domain partition;
    * a **function query stream** ("appearance branch") that is content
      anchored and learns the domain-level function.

The two streams share one residue memory but use *separate* decoders, so each
branch specializes independently.

Crucially, protein function is *domain-level*: a function label only makes sense
relative to a specific segmented domain. So the two branches cannot be fully
independent. Alignment is re-established explicitly at the **function-query
construction step**:

    * each function query is built by pooling the memory over the region
      predicted by its paired segmentation query (mask-conditioned), so
      func query ``i`` literally reads "the residues of the domain seg query
      ``i`` segmented";
    * the shared Hungarian assignment (computed on segmentation masks) still
      binds segmentation query ``i`` and function query ``i`` to the same GT
      domain, so the supervision targets stay aligned.

During training we additionally build **GT-conditioned function queries**
(denoising-style): pool the memory over exact GT domain masks and supervise
their function directly, giving the function branch a stable signal that does
not depend on (initially noisy) predicted masks.

Finally, a fusion module combines the two branches per-domain and pools them
(objectness-weighted) into a **unified protein-level representation**, which is
supervised by the protein-level function label. This unified vector is the
"final protein expression feature" the model learns.

Data flow
---------
    residue_feat [B, L, D]
         |
         +--> shared TransformerEncoder --> memory [B, L, D]
         |
         |   (a) segmentation branch              (b) function branch
         |   domain-aware query selection         mask-conditioned pool + prototypes
         |        |                                        |
         |   seg_queries [B, Q, D]                 func_queries [B, Q, D]
         |        |                                        |
         |   seg_decoder (+ DN queries)             func_decoder (+ GT-func queries)
         |        |                                        |
         |   z_seg [B, Q, D]                       z_func [B, Q, D]
         |        |                                        |
         |   mask head --> mask_logits [B,Q,L]     func head --> func_logits [B,Q,C]
         |        \_____________________  ________________/
         |                              \/
         |                    per-domain fusion [B, Q, rep_dim]
         |                              |
         |                    objectness-weighted pooling
         |                              |
         |                    protein_repr [B, rep_dim]  (unified representation)
         |                              |
         |                    protein_head --> protein_logits [B, C]
"""

import math

import torch
import torch.nn as nn


def sinusoidal_position_encoding(length, dim, device=None):
    """Generate sinusoidal positional encoding [length, dim]."""
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [length, dim]


class DualQueryDecoupledModel(nn.Module):
    def __init__(
        self,
        hidden_dim=1280,
        num_queries=8,
        num_dn_queries=4,
        num_gt_func_queries=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        nhead=8,
        dim_feedforward=2048,
        num_classes=256,
        rep_dim=256,
        task_type="classification",
        function_query_source="mask",
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_dn_queries = num_dn_queries
        self.num_gt_func_queries = num_gt_func_queries
        self.num_classes = num_classes
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        self.rep_dim = rep_dim
        if function_query_source not in {"mask", "global"}:
            raise ValueError(
                "function_query_source must be 'mask' or 'global', "
                f"got {function_query_source!r}"
            )
        self.function_query_source = function_query_source

        # ============ Shared encoder ============
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # ============ Segmentation ("geometry") branch ============
        # Domain-aware scoring: scores each encoder position for query selection.
        self.query_score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Anchor MLP: maps normalized position [0,1] to a position embedding.
        self.anchor_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # Denoising query construction (GT span + noise).
        self.dn_anchor_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # input: [center, width] normalized
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        self.dn_content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True, dropout=0.1,
        )
        self.seg_decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # Mask head with positional encoding (as in QuerySelectionMaskDinoV2).
        self.mask_embed = nn.Linear(hidden_dim, hidden_dim)
        self.mask_gate = nn.Linear(hidden_dim, hidden_dim)
        self.pe_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Domain objectness (matched -> domain, unmatched -> no-domain).
        self.objectness_head = nn.Linear(hidden_dim, 1)

        # ============ Function ("appearance") branch ============
        # Function queries are *mask-conditioned*: their content is the pooled
        # memory over the region predicted by their paired segmentation query.
        # A small learnable prototype provides residual per-slot identity.
        self.func_content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.func_prototype_emb = nn.Embedding(num_queries, hidden_dim)
        nn.init.normal_(self.func_prototype_emb.weight, std=0.02)

        dec_layer_func = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True, dropout=0.1,
        )
        self.func_decoder = nn.TransformerDecoder(dec_layer_func, num_decoder_layers)

        # Function head (domain-level multi-label).
        if task_type == "regression":
            self.func_head = nn.Linear(hidden_dim, 1)
        else:
            self.func_head = nn.Linear(hidden_dim, num_classes)

        # Validity head: predicts whether a domain carries an observed function
        # label (function_label_valid). At inference this lets the model say
        # "this domain's function is not predictable/unknown" instead of
        # fabricating a prediction for unlabeled domains.
        self.validity_head = nn.Linear(hidden_dim, 1)

        # ============ Fusion -> unified protein representation ============
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, rep_dim),
            nn.LayerNorm(rep_dim),
            nn.ReLU(),
        )
        # Supervise the unified representation with the protein-level label.
        self.protein_head = nn.Linear(rep_dim, num_classes)

    def forward(self, residue_feat, residue_mask, gt_spans=None,
                gt_func_targets=None, training=False):
        """
        Args:
            residue_feat: [B, L, D] ESM backbone features.
            residue_mask: [B, L] attention mask (1=valid, 0=pad).
            gt_spans: List[List[Tuple[int, int]]] per-sample GT domain spans
                      (denoising queries only, training).
            gt_func_targets: List[List[Tuple[Tensor[L], Tensor[C]]]] per-sample
                      list of (mask, function-label) for *valid-function* GT
                      domains (GT-conditioned function queries, training).
            training: whether in training mode.

        Returns:
            dict with keys:
                - mask_logits:    [B, Q, L]  (segmentation branch)
                - func_logits:    [B, Q, C]  (function branch; alias class_logits)
                - query_scores:   [B, L]     (selection aux-loss scores)
                - objectness_logits: [B, Q]
                - seg_queries:    [B, Q, D]
                - func_queries:   [B, Q, D]
                - per_domain_repr: [B, Q, rep_dim]
                - protein_repr:   [B, rep_dim]   <- unified representation
                - protein_logits: [B, C]
                - dn_mask_logits / dn_gt_masks (training only)
                - gt_func_logits: [B, K, C] / gt_func_labels: [B, K, C]
                                  (training only)
        """
        B, L, D = residue_feat.shape
        Q = self.num_queries
        pad_mask = (residue_mask == 0)  # True=pad for PyTorch
        device = residue_feat.device

        # ============ Encode ============
        memory = self.encoder(residue_feat, src_key_padding_mask=pad_mask)

        # ============ Segmentation branch ============
        scores = self.query_score_head(memory).squeeze(-1)          # [B, L]
        scores_masked = scores.masked_fill(pad_mask, -1e4)

        _, topk_indices = scores_masked.topk(Q, dim=1)              # [B, Q]
        topk_indices, _ = topk_indices.sort(dim=1)

        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        content_queries = torch.gather(memory, 1, topk_indices_expanded)  # [B, Q, D]

        seq_lens = residue_mask.sum(dim=1, keepdim=True).float()    # [B, 1]
        normalized_positions = topk_indices.float() / seq_lens.clamp(min=1)
        anchor_embed = self.anchor_mlp(normalized_positions.unsqueeze(-1))  # [B, Q, D]

        seg_queries = content_queries + anchor_embed                # [B, Q, D]

        dn_mask_logits = None
        dn_gt_masks = None

        if training and gt_spans is not None and self.num_dn_queries > 0:
            dn_queries, dn_gt_masks = self._build_dn_queries(
                memory, residue_mask, gt_spans, seq_lens
            )                                                       # [B, K, D]
            K = dn_queries.size(1)
            all_queries = torch.cat([seg_queries, dn_queries], dim=1)  # [B, Q+K, D]

            # Isolate DN queries from normal queries in the decoder.
            total_q = Q + K
            tgt_mask = torch.zeros(total_q, total_q, device=device, dtype=torch.bool)
            tgt_mask[Q:, :Q] = True
            tgt_mask[:Q, Q:] = True

            h_all = self.seg_decoder(
                all_queries, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=pad_mask,
            )
            z_seg = h_all[:, :Q, :]
            z_dn = h_all[:, Q:, :]
        else:
            z_seg = self.seg_decoder(
                seg_queries, memory,
                memory_key_padding_mask=pad_mask,
            )

        # Positional-gated mask features.
        pos_enc = sinusoidal_position_encoding(L, D, device=device).unsqueeze(0)  # [1, L, D]
        gate = torch.sigmoid(self.mask_gate(memory))                             # [B, L, D]
        mask_feat = gate * (residue_feat + self.pe_scale * pos_enc)              # [B, L, D]

        mask_logits = torch.einsum(
            "bqd,bld->bql", self.mask_embed(z_seg), mask_feat
        )                                                                        # [B, Q, L]

        if training and gt_spans is not None and self.num_dn_queries > 0:
            dn_mask_logits = torch.einsum(
                "bkd,bld->bkl", self.mask_embed(z_dn), mask_feat
            )                                                                    # [B, K, L]

        objectness_logits = self.objectness_head(z_seg).squeeze(-1)              # [B, Q]

        # ============ Function branch (mask-conditioned) ============
        # Each function query reads the content of the region segmented by its
        # paired segmentation query. This is the explicit region--function
        # alignment: the func query is *derived from* the seg output.
        valid_mask = residue_mask.unsqueeze(1)                                   # [B, 1, L]
        seg_mask_prob = torch.sigmoid(mask_logits) * valid_mask                  # [B, Q, L]
        mask_region_content = (
            seg_mask_prob.unsqueeze(-1) * memory.unsqueeze(1)
        ).sum(dim=2) / seg_mask_prob.sum(dim=2, keepdim=True).clamp(min=1)       # [B, Q, D]

        if self.function_query_source == "mask":
            region_content = mask_region_content
        else:
            # Capacity-matched control: every slot receives the same pooled
            # full-sequence content. The decoder, prototypes, heads, losses,
            # and all segmentation computations remain unchanged.
            global_content = (
                memory * residue_mask.unsqueeze(-1)
            ).sum(dim=1) / residue_mask.sum(dim=1, keepdim=True).clamp(min=1)
            region_content = global_content.unsqueeze(1).expand(-1, Q, -1)

        proto = self.func_prototype_emb(torch.arange(Q, device=device))          # [Q, D]
        func_queries = self.func_content_proj(region_content) + proto.unsqueeze(0)  # [B, Q, D]

        gt_func_logits = None
        gt_func_labels = None
        gt_validity_logits = None
        gt_validity_labels = None

        if training and gt_func_targets is not None and self.num_gt_func_queries > 0:
            gt_func_queries, gt_func_labels, gt_validity_labels = \
                self._build_gt_func_queries(memory, gt_func_targets)            # [B,K,D],[B,K,C],[B,K]
            K = gt_func_queries.size(1)

            all_func_queries = torch.cat([func_queries, gt_func_queries], dim=1)  # [B, Q+K, D]
            total_q = Q + K
            tgt_mask = torch.zeros(total_q, total_q, device=device, dtype=torch.bool)
            tgt_mask[Q:, :Q] = True
            tgt_mask[:Q, Q:] = True

            h_func_all = self.func_decoder(
                all_func_queries, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=pad_mask,
            )
            z_func = h_func_all[:, :Q, :]
            z_gt_func = h_func_all[:, Q:, :]

            gt_func_logits = self.func_head(z_gt_func)                            # [B, K, C]
            gt_validity_logits = self.validity_head(z_gt_func).squeeze(-1)        # [B, K]
        else:
            z_func = self.func_decoder(
                func_queries, memory,
                memory_key_padding_mask=pad_mask,
            )

        if self.task_type == "regression":
            func_logits = self.func_head(z_func)                                 # [B, Q, 1]
        else:
            func_logits = self.func_head(z_func)                                 # [B, Q, C]
        validity_logits = self.validity_head(z_func).squeeze(-1)                 # [B, Q]

        # ============ Fusion -> unified representation ============
        per_domain_repr = self.fusion_mlp(
            torch.cat([z_seg, z_func], dim=-1)
        )                                                                        # [B, Q, rep_dim]

        # Objectness-weighted pooling over domains. Softmax over the objectness
        # logits lets the model emphasize real domains in the final summary.
        pool_weights = torch.softmax(objectness_logits, dim=1)                   # [B, Q]
        protein_repr = torch.einsum(
            "bq,bqd->bd", pool_weights, per_domain_repr
        )                                                                        # [B, rep_dim]

        # Protein-level classification: MAX-POOL over query logits (alignment-
        # free, robust to segmentation quality). Mirrors the previous
        # e2e_protein_cls approach (QuerySelectionMaskDinoV2) that reached
        # protein Fmax 0.907; the objectness-weighted fusion pooling alone was
        # structurally weaker (protein Fmax ~0.52) because it depends on the
        # poorly-trained objectness head. The fused representation is kept as
        # an additive residual via protein_head.
        max_func_logits, _ = func_logits.max(dim=1)                              # [B, C]
        protein_logits = max_func_logits + self.protein_head(protein_repr)       # [B, C]

        result = {
            "mask_logits": mask_logits,
            "func_logits": func_logits,
            "class_logits": func_logits,  # compatibility alias
            "validity_logits": validity_logits,
            "query_scores": scores,
            "objectness_logits": objectness_logits,
            "seg_queries": z_seg,
            "func_queries": z_func,
            "per_domain_repr": per_domain_repr,
            "protein_repr": protein_repr,
            "protein_logits": protein_logits,
        }

        if training and dn_mask_logits is not None:
            result["dn_mask_logits"] = dn_mask_logits
            result["dn_gt_masks"] = dn_gt_masks

        if training and gt_func_logits is not None:
            result["gt_func_logits"] = gt_func_logits
            result["gt_func_labels"] = gt_func_labels
            result["gt_validity_logits"] = gt_validity_logits
            result["gt_validity_labels"] = gt_validity_labels

        return result

    def _build_dn_queries(self, memory, residue_mask, gt_spans, seq_lens):
        """Build denoising queries from GT spans with noise (identical to V2)."""
        B, L, D = memory.shape
        K = self.num_dn_queries
        device = memory.device

        dn_queries_list = []
        dn_gt_masks_list = []

        for b in range(B):
            spans = gt_spans[b]
            seq_len = int(seq_lens[b].item())

            if len(spans) == 0:
                dn_q = torch.zeros(K, D, device=device)
                dn_gt = [torch.zeros(L, device=device) for _ in range(K)]
            else:
                dn_q_list = []
                dn_gt_list = []

                for k in range(K):
                    span_idx = torch.randint(0, len(spans), (1,)).item()
                    start, end = spans[span_idx]

                    center = (start + end) / 2.0
                    width = end - start

                    center_noise = (torch.rand(1, device=device).item() - 0.5) * 0.2 * seq_len
                    scale_noise = 0.7 + torch.rand(1, device=device).item() * 0.6

                    noisy_center = max(0, min(seq_len - 1, center + center_noise))
                    noisy_width = max(5, min(seq_len, width * scale_noise))

                    norm_center = noisy_center / max(seq_len, 1)
                    norm_width = noisy_width / max(seq_len, 1)
                    anchor_input = torch.tensor(
                        [norm_center, norm_width], device=device, dtype=torch.float32
                    )
                    anchor_emb = self.dn_anchor_mlp(anchor_input.unsqueeze(0))

                    noisy_start = int(max(0, noisy_center - noisy_width / 2))
                    noisy_end = int(min(seq_len, noisy_center + noisy_width / 2))
                    if noisy_end <= noisy_start:
                        noisy_end = noisy_start + 1
                    content = memory[b, noisy_start:noisy_end, :].mean(dim=0, keepdim=True)
                    content = self.dn_content_proj(content)

                    dn_q_list.append(content + anchor_emb)

                    gt_mask = torch.zeros(L, device=device)
                    gt_mask[start:end] = 1.0
                    dn_gt_list.append(gt_mask)

                dn_q = torch.cat(dn_q_list, dim=0)
                dn_gt = dn_gt_list

            dn_queries_list.append(dn_q)
            dn_gt_masks_list.append(dn_gt)

        dn_queries = torch.stack(dn_queries_list, dim=0)
        return dn_queries, dn_gt_masks_list

    def _build_gt_func_queries(self, memory, gt_func_targets):
        """
        Build ground-truth-conditioned function queries (denoising-style).

        Each query pools the memory over an exact GT domain mask and is
        supervised directly by that domain's function label (no Hungarian
        matching). This gives the function branch a stable signal that does not
        depend on the (initially noisy) predicted masks.

        Args:
            memory: [B, L, D]
            gt_func_targets: List[List[(mask[L], label[C], valid)]] per-sample
                             list of ALL GT domains (valid + invalid; invalid
                             domains carry an all-zero label and valid=False).

        Returns:
            gt_func_queries: [B, K, D]
            gt_func_labels:  [B, K, C]
            gt_validity_labels: [B, K]  (1 if the sampled domain is
                                 function_label_valid, else 0)
        """
        B, L, D = memory.shape
        K = self.num_gt_func_queries
        C = self.num_classes
        device = memory.device

        queries_list = []
        labels_list = []
        validity_list = []

        for b in range(B):
            targets = gt_func_targets[b]

            if len(targets) == 0:
                queries_list.append(torch.zeros(K, D, device=device))
                labels_list.append(torch.zeros(K, C, device=device))
                validity_list.append(torch.zeros(K, device=device))
                continue

            qs = []
            labs = []
            vls = []
            for k in range(K):
                idx = torch.randint(0, len(targets), (1,)).item()
                mask, label, valid = targets[idx]
                m = mask.to(device).unsqueeze(-1)                                # [L, 1]
                content = (memory[b] * m).sum(dim=0) / m.sum().clamp(min=1)      # [D]
                qs.append(self.func_content_proj(content))                       # [D]
                labs.append(label.to(device))                                    # [C]
                vls.append(1.0 if valid else 0.0)
            queries_list.append(torch.stack(qs, dim=0))                          # [K, D]
            labels_list.append(torch.stack(labs, dim=0))                         # [K, C]
            validity_list.append(torch.tensor(vls, device=device))               # [K]

        gt_func_queries = torch.stack(queries_list, dim=0)                       # [B, K, D]
        gt_func_labels = torch.stack(labels_list, dim=0)                         # [B, K, C]
        gt_validity_labels = torch.stack(validity_list, dim=0)                   # [B, K]
        return gt_func_queries, gt_func_labels, gt_validity_labels
