"""
Lightning module for end-to-end domain segmentation + protein-level classification.

Architecture: QuerySelectionMaskDinoV2 (strictly aligned with hemenet/query_v2).
Task loss: all query features → max-pool → protein-level multi-label focal loss.

Loss formula:
    total_loss = mask_loss + dn_loss_weight * dn_loss 
                + domain_sel_loss_weight * domain_sel_loss 
                + task_loss_weight * task_loss
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

from model import QuerySelectionMaskDinoV2
from loss import (
    DomainHungarianMatcher,
    MultiDomainMaskLoss,
    DenoisingMaskLoss,
    DomainSelectionLoss,
    ProteinClassificationLoss,
)


def compute_fmax(preds, labels, thresholds=None):
    """Compute Fmax metric for multi-label classification."""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    fmax, best_t = 0, 0
    for t in thresholds:
        pred_bin = (preds >= t).astype(int)
        prec_list, rec_list = [], []
        for i in range(labels.shape[0]):
            tp = np.sum(pred_bin[i] * labels[i])
            if np.sum(pred_bin[i]) > 0:
                prec_list.append(tp / np.sum(pred_bin[i]))
            if np.sum(labels[i]) > 0:
                rec_list.append(tp / np.sum(labels[i]))
        prec = np.mean(prec_list) if prec_list else 0
        rec = np.mean(rec_list) if rec_list else 0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f > fmax:
            fmax, best_t = f, t
    return fmax, best_t


class E2EProteinClsLightning(pl.LightningModule):
    def __init__(
        self,
        num_classes=256,
        num_queries=8,
        num_dn_queries=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        lr=1e-4,
        task_loss_weight=50.0,
        dn_loss_weight=1.0,
        domain_sel_loss_weight=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = QuerySelectionMaskDinoV2(
            num_classes=num_classes,
            num_queries=num_queries,
            num_dn_queries=num_dn_queries,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            task_type="classification",
        )

        # Loss functions
        self.matcher = DomainHungarianMatcher()
        self.mask_loss_fn = MultiDomainMaskLoss()
        self.dn_loss_fn = DenoisingMaskLoss()
        self.domain_selection_loss_fn = DomainSelectionLoss()
        self.task_loss_fn = ProteinClassificationLoss()

        # Fixed loss weights
        self.task_loss_weight = task_loss_weight
        self.dn_loss_weight = dn_loss_weight
        self.domain_sel_loss_weight = domain_sel_loss_weight

        # Validation buffers
        self.val_preds = []
        self.val_labels = []
        self.val_mask_logits = []
        self.val_domain_masks = []
        self.val_seq_lens = []

    def training_step(self, batch, batch_idx):
        residue_feat = batch["residue_feat"].to(self.device)   # [B, L, D]
        attn_mask = batch["attn_mask"].to(self.device)         # [B, L]
        domain_mask = batch["domain_mask"].to(self.device)     # [B, L]
        protein_label = batch["protein_label"].to(self.device) # [B, C]
        domain_infos = batch["domain_infos"]                   # List[List[dict]]

        B = residue_feat.size(0)

        # Extract GT spans for denoising training
        gt_spans = self._extract_gt_spans(domain_infos, attn_mask)

        # Forward with denoising
        outputs = self.model(residue_feat, attn_mask, gt_spans=gt_spans, training=True)

        pred_masks = outputs["mask_logits"]     # [B, Q, L]
        query_scores = outputs["query_scores"]  # [B, L]

        # ============ Loss 1: Hungarian-matched mask loss ============
        total_mask_loss = 0.0
        for b in range(B):
            gt_masks_b = [d["mask"].to(self.device) for d in domain_infos[b]]
            if len(gt_masks_b) == 0:
                continue
            matches = self.matcher.match(pred_masks[b], gt_masks_b)
            total_mask_loss += self.mask_loss_fn(pred_masks[b], gt_masks_b, matches)
        total_mask_loss /= B

        # ============ Loss 2: Protein-level classification loss ============
        task_loss = self.task_loss_fn(outputs["class_logits"], protein_label)

        # ============ Loss 3: Denoising mask loss ============
        if "dn_mask_logits" in outputs and outputs["dn_mask_logits"] is not None:
            dn_loss = self.dn_loss_fn(outputs["dn_mask_logits"], outputs["dn_gt_masks"])
        else:
            dn_loss = torch.tensor(0.0, device=self.device)

        # ============ Loss 4: Domain-aware selection loss ============
        domain_sel_loss = self.domain_selection_loss_fn(
            query_scores, domain_mask, attn_mask
        )

        # ============ Fixed-weight total loss ============
        total_loss = (
            total_mask_loss
            + self.dn_loss_weight * dn_loss
            + self.domain_sel_loss_weight * domain_sel_loss
            + self.task_loss_weight * task_loss
        )

        self.log_dict({
            "train_loss": total_loss,
            "mask_loss": total_mask_loss,
            "task_loss": task_loss,
            "dn_loss": dn_loss,
            "domain_sel_loss": domain_sel_loss,
        }, prog_bar=False, batch_size=B)

        return total_loss

    def validation_step(self, batch, batch_idx):
        residue_feat = batch["residue_feat"].to(self.device)
        attn_mask = batch["attn_mask"].to(self.device)
        protein_label = batch["protein_label"].to(self.device)
        domain_mask = batch["domain_mask"]
        seq_lens = batch["seq_len"]

        with torch.no_grad():
            outputs = self.model(residue_feat, attn_mask, gt_spans=None, training=False)

        # Classification: max-pool over queries
        logits = outputs["class_logits"]  # [B, Q, C]
        seq_logits, _ = logits.max(dim=1)  # [B, C]
        probs = torch.sigmoid(seq_logits)

        self.val_preds.append(probs.detach().cpu())
        self.val_labels.append(protein_label.detach().cpu())

        # Domain segmentation data
        mask_logits = outputs["mask_logits"]  # [B, Q, L]
        for b in range(mask_logits.size(0)):
            self.val_mask_logits.append(mask_logits[b].detach().cpu())
            self.val_domain_masks.append(domain_mask[b].detach().cpu())
            self.val_seq_lens.append(seq_lens[b])

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return

        preds = torch.cat(self.val_preds, dim=0)
        labels = torch.cat(self.val_labels, dim=0)

        # Fmax
        fmax, best_t = compute_fmax(preds, labels)
        self.log("val_Fmax", fmax, prog_bar=True)
        self.log("best_threshold", best_t)

        # F1 at 0.5
        preds_bin = (preds > 0.5).int()
        f1 = f1_score(
            labels.numpy(), preds_bin.numpy(),
            average="micro", zero_division=0,
        )
        self.log("val_f1", f1, prog_bar=True)

        # Domain segmentation IoU
        val_domain_iou = self._compute_domain_iou()
        self.log("val_domain_IoU", val_domain_iou, prog_bar=True)

        # Cleanup
        self.val_preds.clear()
        self.val_labels.clear()
        self.val_mask_logits.clear()
        self.val_domain_masks.clear()
        self.val_seq_lens.clear()

    def _compute_domain_iou(self):
        """Compute domain segmentation IoU using NMS+merge post-processing."""
        from evaluate_domain import apply_nms_merge_to_mask_logits, evaluate_single_protein

        ious = []
        weights = []

        for i in range(len(self.val_mask_logits)):
            seq_len = self.val_seq_lens[i]
            gt_mask = self.val_domain_masks[i]

            # Convert binary mask to domain labels
            gt_labels = self._binary_mask_to_domain_labels(gt_mask, seq_len)
            if gt_labels.max() == 0:
                continue

            pred_labels = apply_nms_merge_to_mask_logits(
                self.val_mask_logits[i], seq_len
            )
            metrics = evaluate_single_protein(pred_labels, gt_labels, tolerance=20)
            domain_weight = max(np.sum(gt_labels > 0), 1)
            ious.append(metrics["iou"])
            weights.append(domain_weight)

        if len(ious) > 0:
            w = np.array(weights, dtype=float)
            w /= w.sum() + 1e-8
            return float(np.average(ious, weights=w))
        return 0.0

    def _binary_mask_to_domain_labels(self, mask, seq_len):
        """Convert binary domain mask [L] to per-residue domain ID labels [seq_len]."""
        labels = np.zeros(seq_len, dtype=int)
        m = mask[:seq_len].numpy()
        domain_id = 0
        in_span = False
        for i in range(seq_len):
            if m[i] > 0.5 and not in_span:
                in_span = True
                domain_id += 1
            elif m[i] <= 0.5:
                in_span = False
            if in_span:
                labels[i] = domain_id
        return labels

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _extract_gt_spans(self, domain_infos, attn_mask):
        """
        Extract GT domain spans from domain_infos for denoising training.

        Args:
            domain_infos: List[List[dict]] per-sample domain info
            attn_mask: [B, L] attention mask

        Returns:
            List[List[Tuple[int, int]]] per-sample list of (start, end) spans
        """
        B = len(domain_infos)
        L = attn_mask.size(1)
        all_spans = []

        for b in range(B):
            spans = []
            for d_info in domain_infos[b]:
                mask = d_info["mask"]  # [L]
                # Extract contiguous spans from binary mask
                in_span = False
                start = 0
                for i in range(L):
                    if mask[i] > 0.5 and not in_span:
                        in_span = True
                        start = i
                    elif mask[i] <= 0.5 and in_span:
                        spans.append((start, i))
                        in_span = False
                if in_span:
                    seq_len = int(attn_mask[b].sum().item())
                    spans.append((start, min(i + 1, seq_len)))
            all_spans.append(spans)

        return all_spans
