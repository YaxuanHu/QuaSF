"""Continue pretrained QuaSF with a regional function-query curriculum."""
from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F

from .lightning_module import DualQueryLightning, compute_fmax
from .metrics import matches as unified_matches
from .model import BatchedJointFunctionQuery, DirectFunctionQuery


class JointCurriculum(DualQueryLightning):
    """Reuse the original validation protocol with a new joint training objective."""

    def __init__(self, source_hparams: dict, mask_threshold: float = 0.3,
                 query_warmup_epochs: int = 1, seg_lr: float = 1e-5,
                 query_lr: float = 3e-5) -> None:
        super().__init__(**source_hparams)
        self.query_warmup_epochs = query_warmup_epochs
        self.seg_lr = seg_lr
        self.query_lr = query_lr
        self.joint_mask_threshold = mask_threshold
        self.val_region_predictions: list[torch.Tensor] = []
        self.val_region_targets: list[torch.Tensor] = []

    def attach_query(self, query: DirectFunctionQuery) -> None:
        """Replace the module boundary while retaining the original validator."""
        self.model = BatchedJointFunctionQuery(self.model, query, self.joint_mask_threshold)
        for name, parameter in self.model.segmenter.named_parameters():
            parameter.requires_grad_(name.startswith(
                ("seg_decoder.", "mask_embed.", "mask_gate.", "pe_scale")))

    def on_train_epoch_start(self) -> None:
        """Warm up functions with a zero segmentation LR, preserving DDP buckets."""
        joint = self.current_epoch >= self.query_warmup_epochs
        self.trainer.optimizers[0].param_groups[0]["lr"] = self.seg_lr if joint else 0.0
        self.log("joint_stage", float(joint), sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Keep the segmentation learning rate smaller than the function rate."""
        segmenter = [parameter for name, parameter in self.model.segmenter.named_parameters()
                     if name.startswith(("seg_decoder.", "mask_embed.",
                                         "mask_gate.", "pe_scale"))]
        function = [parameter for parameter in self.model.function_query.parameters()
                    if parameter.requires_grad]
        return torch.optim.AdamW([{"params": segmenter, "lr": self.seg_lr},
                                  {"params": function, "lr": self.query_lr}],
                                 weight_decay=1e-2)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Supervise matched masks, their distinct labels, and protein labels."""
        features = batch["residue_feat"].to(self.device)
        residue_mask = batch["attn_mask"].to(self.device)
        outputs = self.model(features, residue_mask)
        masks = outputs["mask_logits"]
        logits = outputs["region_logits"]
        mask_terms: list[torch.Tensor] = []
        function_terms: list[torch.Tensor] = []
        rank_terms: list[torch.Tensor] = []
        for index, domains in enumerate(batch["domain_infos"]):
            gt_masks = [item["mask"].to(self.device) for item in domains]
            matches = self.matcher.match(masks[index], gt_masks,
                                         valid_mask=residue_mask[index].bool())
            if gt_masks:
                union = torch.stack(gt_masks).amax(0)
                assigned = dict(matches)
                for query_index in range(masks.shape[1]):
                    target = gt_masks[assigned[query_index]] if query_index in assigned else union
                    mask_terms.append(self._balanced_bce_dice(
                        masks[index, query_index], target, residue_mask[index].bool()))
            # Final decoded regions may share one source query. Match those
            # regions independently, rather than reusing the query assignment.
            decoded = outputs["region_masks"][index]
            candidate = [j for j in range(len(decoded))
                         if bool(outputs["region_valid"][index, j])]
            region_matches = self.matcher.match(
                torch.where(decoded[candidate], 20.0, -20.0), gt_masks,
                valid_mask=residue_mask[index].bool()) if candidate else []
            labelled: list[tuple[int, torch.Tensor]] = []
            for position, gt_index in region_matches:
                query_index = candidate[position]
                domain = domains[gt_index]
                if not domain["valid"]:
                    continue
                intersection = (decoded[query_index] & gt_masks[gt_index].bool()).sum().float()
                union_size = (decoded[query_index] | gt_masks[gt_index].bool()).sum().clamp_min(1)
                if float(intersection / union_size) < 0.5:
                    continue
                label = domain["label"].to(self.device)
                positive_weight = torch.where(label > 0.5, 12.0, 1.0)
                function_terms.append(F.binary_cross_entropy_with_logits(
                    logits[index, query_index], label, weight=positive_weight))
                labelled.append((query_index, label))
            for position, (left, left_label) in enumerate(labelled):
                for right, right_label in labelled[position + 1:]:
                    left_only = (left_label > 0.5) & (right_label < 0.5)
                    right_only = (right_label > 0.5) & (left_label < 0.5)
                    if left_only.any():
                        rank_terms.append(F.softplus(2.0 - logits[index, left, left_only]
                                                     + logits[index, right, left_only]).mean())
                    if right_only.any():
                        rank_terms.append(F.softplus(2.0 - logits[index, right, right_only]
                                                     + logits[index, left, right_only]).mean())
        zero = logits.sum() * 0
        mask_loss = torch.stack(mask_terms).mean() if mask_terms else zero
        function_loss = torch.stack(function_terms).mean() if function_terms else zero
        rank_loss = torch.stack(rank_terms).mean() if rank_terms else zero
        protein_loss = self.rep_loss_fn(outputs["protein_logits"],
                                        batch["protein_label"].to(self.device))
        loss = 1.5 * mask_loss + 10.0 * function_loss + 0.5 * rank_loss + protein_loss
        self.log_dict({"train_loss": loss, "joint_mask_loss": mask_loss,
                       "joint_function_loss": function_loss,
                       "joint_rank_loss": rank_loss,
                       "joint_protein_loss": protein_loss},
                      batch_size=len(features), sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Retain historical metrics and add strict final-region function scoring."""
        super().validation_step(batch, batch_idx)
        outputs = self.model(batch["residue_feat"].to(self.device),
                             batch["attn_mask"].to(self.device), training=False)
        for index, domains in enumerate(batch["domain_infos"]):
            actual = [j for j, item in enumerate(domains) if item["valid"]]
            gt_masks = [item["mask"].to(self.device) for item in domains]
            decoded = outputs["region_masks"][index]
            candidates = [j for j in range(len(decoded))
                          if bool(outputs["region_valid"][index, j])]
            matrix = np.asarray([
                [float((decoded[j] & gt.bool()).sum()) /
                 max(1, int((decoded[j] | gt.bool()).sum())) for gt in gt_masks]
                for j in candidates], dtype=float).reshape(len(candidates), len(gt_masks))
            matches = unified_matches(matrix, 0.5)
            used_predictions: set[int] = set()
            used_targets: set[int] = set()
            for position, gt_index, _ in matches:
                pred_index = candidates[position]
                if gt_index not in actual:
                    continue
                self.val_region_predictions.append(outputs["region_logits"][index, pred_index].sigmoid().cpu())
                self.val_region_targets.append(domains[gt_index]["label"].float().cpu())
                used_predictions.add(pred_index)
                used_targets.add(gt_index)
            zeros = torch.zeros(256)
            for gt_index in actual:
                if gt_index not in used_targets:
                    self.val_region_predictions.append(zeros)
                    self.val_region_targets.append(domains[gt_index]["label"].float().cpu())
            for pred_index in candidates:
                if pred_index not in used_predictions:
                    self.val_region_predictions.append(outputs["region_logits"][index, pred_index].sigmoid().cpu())
                    self.val_region_targets.append(zeros)

    def on_validation_epoch_end(self) -> None:
        """Select with region-level strict Fmax, then log legacy metrics."""
        if self.val_region_predictions:
            predictions = torch.stack(self.val_region_predictions).to(self.device)
            targets = torch.stack(self.val_region_targets).to(self.device)
        else:
            predictions = torch.zeros((0, 256), device=self.device)
            targets = torch.zeros((0, 256), device=self.device)
        predictions = self._gather_variable_rows(predictions).cpu()
        targets = self._gather_variable_rows(targets).cpu()
        strict_fmax, threshold = compute_fmax(predictions, targets)
        self.log("val_region_strict_Fmax", strict_fmax, sync_dist=True, prog_bar=True)
        self.log("val_region_threshold", threshold, sync_dist=True)
        self.val_region_predictions.clear()
        self.val_region_targets.clear()
        super().on_validation_epoch_end()
