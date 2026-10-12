"""
Lightning module for the dual-query decoupled region--function model.

Architecture: DualQueryDecoupledModel (QuerySplat-inspired decoupling).
    * segmentation query stream -> domain masks (Hungarian-matched)
    * function query stream    -> domain-level function (same assignment)
    * fusion                   -> unified protein representation
                                 (supervised by the protein-level label)

Loss formula:
    total_loss = mask_loss
               + dn_loss_weight * dn_loss
               + domain_sel_loss_weight * domain_sel_loss
               + task_loss_weight * task_loss          (domain-level function)
               + objectness_loss_weight * objectness_loss
               + rep_loss_weight * rep_loss            (unified representation)
"""

import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from model import DualQueryDecoupledModel
from loss import (
    DomainHungarianMatcher,
    MultiDomainMaskLoss,
    DenoisingMaskLoss,
    DomainSelectionLoss,
    DomainClassificationLoss,
    ProteinRepresentationLoss,
    GtConditionedFunctionLoss,
)


def compute_fmax(preds, labels, thresholds=None):
    """Compute Fmax metric for multi-label classification."""
    preds = preds.detach().float().cpu().numpy()
    labels = labels.detach().float().cpu().numpy()
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


class DualQueryLightning(pl.LightningModule):
    def __init__(
        self,
        num_classes=256,
        num_queries=8,
        num_dn_queries=4,
        num_gt_func_queries=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        rep_dim=256,
        lr=1e-4,
        task_loss_weight=30.0,
        rep_loss_weight=5.0,
        gt_func_loss_weight=10.0,
        dn_loss_weight=1.0,
        domain_sel_loss_weight=0.5,
        unmatched_mask_weight=0.1,
        objectness_loss_weight=1.0,
        mask_loss_weight=1.5,
        validity_loss_weight=0.3,
        mask_threshold=0.5,
        objectness_threshold=0.3,
        # v2 fix: curriculum + split LR + warmup
        curriculum_coarse_end=12,
        curriculum_ramp_epochs=5,
        curriculum_freeze_encoder=True,
        lr_head=3e-4,
        lr_enc=1e-4,
        warmup_ratio=0.05,
        function_query_source="mask",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DualQueryDecoupledModel(
            num_classes=num_classes,
            num_queries=num_queries,
            num_dn_queries=num_dn_queries,
            num_gt_func_queries=num_gt_func_queries,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            rep_dim=rep_dim,
            task_type="classification",
            function_query_source=function_query_source,
        )

        # Loss functions
        self.matcher = DomainHungarianMatcher()
        self.mask_loss_fn = MultiDomainMaskLoss()
        self.dn_loss_fn = DenoisingMaskLoss()
        self.domain_selection_loss_fn = DomainSelectionLoss()
        self.task_loss_fn = DomainClassificationLoss()
        self.rep_loss_fn = ProteinRepresentationLoss()
        self.gt_func_loss_fn = GtConditionedFunctionLoss()

        # Loss weights
        self.task_loss_weight = task_loss_weight
        self.rep_loss_weight = rep_loss_weight
        self.gt_func_loss_weight = gt_func_loss_weight
        self.dn_loss_weight = dn_loss_weight
        self.domain_sel_loss_weight = domain_sel_loss_weight
        self.unmatched_mask_weight = unmatched_mask_weight
        self.objectness_loss_weight = objectness_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.validity_loss_weight = validity_loss_weight

        # Validation buffers
        self.val_domain_preds = []
        self.val_domain_labels = []
        self.val_e2e_domain_preds = []
        self.val_e2e_domain_labels = []
        self.val_e2e_matched_preds = []   # 口径B: matched GT domains only
        self.val_e2e_matched_labels = []
        self.val_e2e_validity_preds = []  # matched domains: validity logits
        self.val_e2e_validity_targets = []  # matched domains: GT valid flag
        self.val_e2e_gt_count = 0
        self.val_e2e_match_count = 0
        self.val_e2e_extra_count = 0
        self.val_protein_preds = []
        self.val_protein_labels = []
        self.val_mask_logits = []
        self.val_gt_labels = []
        self.val_objectness_logits = []
        self.val_seq_lens = []

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        residue_feat = batch["residue_feat"].to(self.device)
        attn_mask = batch["attn_mask"].to(self.device)
        domain_mask = batch["domain_mask"].to(self.device)
        protein_label = batch["protein_label"].to(self.device)
        domain_infos = batch["domain_infos"]

        B = residue_feat.size(0)

        gt_spans = self._extract_gt_spans(domain_infos, attn_mask)
        gt_func_targets = [
            [(d["mask"], d["label"], bool(d["valid"])) for d in domain_infos[b]]
            for b in range(B)
        ]
        outputs = self.model(
            residue_feat, attn_mask, gt_spans=gt_spans,
            gt_func_targets=gt_func_targets, training=True,
        )

        pred_masks = outputs["mask_logits"]
        func_logits = outputs["func_logits"]
        query_scores = outputs["query_scores"]
        protein_logits = outputs["protein_logits"]

        # Loss 1: mask loss (segmentation branch).
        # Coarse phase: EVERY query learns the union GT domain mask with no
        # Hungarian matching. This mirrors the v3 curriculum coarse phase that
        # reached IoU 0.888 stably; per-query Hungarian matching (which leaves
        # 7-8 of 10 queries with pure-negative targets and collapses mask
        # probabilities) is deferred to the precise phase.
        coarse_phase = float(self.current_epoch) < float(
            self.hparams.curriculum_coarse_end
        )
        total_mask_loss = pred_masks.sum() * 0.0
        total_objectness_loss = pred_masks.sum() * 0.0
        matches_list = []
        if coarse_phase:
            for b in range(B):
                valid_mask_b = attn_mask[b].bool()
                union_gt = torch.zeros_like(attn_mask[b].float())
                for d in domain_infos[b]:
                    union_gt = torch.maximum(
                        union_gt, d["mask"].float().to(self.device)
                    )
                for q in range(pred_masks.size(1)):
                    total_mask_loss += self._balanced_bce_dice(
                        pred_masks[b, q], union_gt, valid_mask_b
                    )
                matches_list.append([])
                obj_target = torch.ones_like(outputs["objectness_logits"][b])
                total_objectness_loss += self._balanced_objectness_loss(
                    outputs["objectness_logits"][b], obj_target
                )
            total_mask_loss /= B * pred_masks.size(1)
            total_objectness_loss /= B
        else:
            for b in range(B):
                gt_masks_b = [d["mask"].to(self.device) for d in domain_infos[b]]
                valid_mask_b = attn_mask[b].bool()
                matches = self.matcher.match(
                    pred_masks[b], gt_masks_b, valid_mask=valid_mask_b
                )
                matches_list.append(matches)
                # Precise phase: matched queries learn their own GT domain mask;
                # UNMATCHED queries keep learning the union domain mask instead
                # of being pushed to all-zero. Suppressing unmatched queries to
                # background collapses the predicted mask probabilities (IoU
                # 0.87 -> 0.20 in smoke4); keeping them as domain predictors
                # preserves segmentation while matched queries specialize.
                union_gt = torch.zeros_like(attn_mask[b].float())
                for d in domain_infos[b]:
                    union_gt = torch.maximum(
                        union_gt, d["mask"].float().to(self.device)
                    )
                matched_qs = {q for q, _ in matches}
                for q in range(pred_masks.size(1)):
                    if q in matched_qs:
                        gt_idx = next(g for qq, g in matches if qq == q)
                        target = gt_masks_b[gt_idx]
                    else:
                        target = union_gt
                    total_mask_loss += self._balanced_bce_dice(
                        pred_masks[b, q], target, valid_mask_b
                    )
                objectness_target = torch.zeros_like(outputs["objectness_logits"][b])
                for q_idx, _ in matches:
                    objectness_target[q_idx] = 1.0
                total_objectness_loss += self._balanced_objectness_loss(
                    outputs["objectness_logits"][b], objectness_target
                )
            total_mask_loss /= B * pred_masks.size(1)
            total_objectness_loss /= B

        # Loss 2: domain-level function loss (function branch, shared assignment).
        task_loss = self.task_loss_fn(
            func_logits, matches_list, domain_infos, self.device
        )

        # Loss 3: denoising mask loss (segmentation branch).
        if "dn_mask_logits" in outputs and outputs["dn_mask_logits"] is not None:
            dn_loss = self.dn_loss_fn(outputs["dn_mask_logits"], outputs["dn_gt_masks"])
        else:
            dn_loss = torch.tensor(0.0, device=self.device)

        # Loss 4: domain-aware selection loss (segmentation branch).
        domain_sel_loss = self.domain_selection_loss_fn(
            query_scores, domain_mask, attn_mask
        )

        # Loss 5: unified protein-representation loss.
        rep_loss = self.rep_loss_fn(protein_logits, protein_label)

        # Loss 5b: domain validity (does this domain carry an observed function?).
        # Supervised via GT-conditioned queries from epoch 0 (stable, no
        # Hungarian-matching noise): GT-conditioned queries pool exact GT domain
        # masks (valid + invalid) and predict function_label_valid. This is the
        # v6 design - the v5 approach (matched-query validity from ep12) failed
        # (validity_acc collapsed to chance and dragged main metrics down).
        if outputs.get("gt_validity_logits") is not None:
            gv_logits = outputs["gt_validity_logits"]
            gv_targets = outputs["gt_validity_labels"]
            gv_bce = F.binary_cross_entropy_with_logits(
                gv_logits, gv_targets, reduction="none"
            )
            pos = gv_targets > 0.5
            neg = ~pos
            terms = []
            if pos.any():
                terms.append(gv_bce[pos].mean())
            if neg.any():
                terms.append(gv_bce[neg].mean())
            validity_loss = torch.stack(terms).mean() if terms else gv_bce.mean()
        else:
            validity_loss = torch.tensor(0.0, device=self.device)

        # Loss 6: GT-conditioned function loss (denoising-style direct supervision).
        if "gt_func_logits" in outputs and outputs["gt_func_logits"] is not None:
            gt_func_loss = self.gt_func_loss_fn(
                outputs["gt_func_logits"], outputs["gt_func_labels"]
            )
        else:
            gt_func_loss = torch.tensor(0.0, device=self.device)

        # Curriculum: coarse phase gates the DOMAIN-level classification losses.
        # The protein-level loss (max-pooled focal) is ALWAYS on: it does not
        # depend on segmentation quality and mirrors the e2e_protein_cls recipe
        # (task_loss_weight=50 from epoch 0) that reached protein Fmax 0.907.
        eff_task_w, eff_gt_w, _ = self._curriculum_weights()

        total_loss = (
            self.mask_loss_weight * total_mask_loss
            + self.dn_loss_weight * dn_loss
            + self.domain_sel_loss_weight * domain_sel_loss
            + eff_task_w * task_loss
            + self.objectness_loss_weight * total_objectness_loss
            + self.rep_loss_weight * rep_loss
            + eff_gt_w * gt_func_loss
            + self.validity_loss_weight * validity_loss
        )

        self.log_dict({
            "train_loss": total_loss,
            "mask_loss": total_mask_loss,
            "task_loss": task_loss,
            "rep_loss": rep_loss,
            "gt_func_loss": gt_func_loss,
            "dn_loss": dn_loss,
            "domain_sel_loss": domain_sel_loss,
            "objectness_loss": total_objectness_loss,
            "validity_loss": validity_loss,
            "effective_task_weight": eff_task_w,
        }, prog_bar=False, batch_size=B)

        return total_loss

    def _curriculum_weights(self):
        """Return (task_w, gt_func_w, rep_w) gated by the coarse->precise schedule.

        - epoch < curriculum_coarse_end: segmentation-only phase, all
          classification losses off (weight 0).
        - afterwards: linear ramp-in over curriculum_ramp_epochs up to the
          configured base weights.
        """
        ep = float(self.current_epoch)
        coarse_end = float(self.hparams.curriculum_coarse_end)
        ramp = max(1.0, float(self.hparams.curriculum_ramp_epochs))
        if ep < coarse_end:
            return 0.0, 0.0, 0.0
        p = min(1.0, (ep - coarse_end + 1.0) / ramp)
        return (
            self.hparams.task_loss_weight * p,
            self.hparams.gt_func_loss_weight * p,
            self.hparams.rep_loss_weight * p,
        )

    def on_train_epoch_start(self):
        """Classification phase: freeze the shared encoder.

        The smoke runs showed mask probabilities collapse once classification
        gradients flow through the shared encoder (IoU 0.53 -> 0.09-0.22 while
        mask_loss kept decreasing). Freezing the encoder during the
        classification phase keeps the segmentation representation stable; only
        the query decoders and heads continue training.
        """
        if not getattr(self.hparams, "curriculum_freeze_encoder", True):
            return
        freeze = float(self.current_epoch) >= float(
            getattr(self.hparams, "curriculum_coarse_end", 12)
        )
        for p in self.model.encoder.parameters():
            p.requires_grad_(not freeze)

    @staticmethod
    def _balanced_bce_dice(pred, gt, valid_mask):
        """Balanced BCE + Dice between one query's mask and a target mask.

        BCE is averaged over in-domain and out-of-domain pixels separately so
        the large background does not collapse predicted probabilities.
        """
        pred = pred[valid_mask]
        gt = gt[valid_mask]
        bce_all = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
        pos = gt > 0.5
        neg = ~pos
        terms = []
        if pos.any():
            terms.append(bce_all[pos].mean())
        if neg.any():
            terms.append(bce_all[neg].mean())
        bce = torch.stack(terms).mean() if terms else bce_all.mean()
        prob = pred.sigmoid()
        dice = 1.0 - (2.0 * (prob * gt).sum() + 1e-6) / (
            prob.sum() + gt.sum() + 1e-6
        )
        return bce + dice

    @staticmethod
    def _balanced_objectness_loss(logits, targets):
        """Give positive and negative queries equal aggregate weight.

        A plain mean BCE is dominated by unmatched queries when only a small
        fraction of queries correspond to domains, which admits the observed
        all-negative objectness collapse.  Averaging each class separately
        keeps the loss scale stable while preserving both gradients.
        """
        per_query = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        positive = targets > 0.5
        negative = ~positive
        terms = []
        if positive.any():
            terms.append(per_query[positive].mean())
        if negative.any():
            terms.append(per_query[negative].mean())
        return torch.stack(terms).mean()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        residue_feat = batch["residue_feat"].to(self.device)
        attn_mask = batch["attn_mask"].to(self.device)
        domain_infos = batch["domain_infos"]
        seq_lens = batch["seq_len"]

        with torch.no_grad():
            outputs = self.model(residue_feat, attn_mask, gt_spans=None, training=False)

        B = residue_feat.size(0)
        pred_masks = outputs["mask_logits"]
        func_logits = outputs["func_logits"]
        protein_logits = outputs["protein_logits"]

        # Per-domain function evaluation (oracle-aligned, function branch).
        for b in range(B):
            gt_masks_b = [d["mask"].to(self.device) for d in domain_infos[b]]
            if len(gt_masks_b) == 0:
                continue
            matches = self.matcher.match(
                pred_masks[b], gt_masks_b, valid_mask=attn_mask[b].bool()
            )
            for q_idx, gt_idx in matches:
                if not domain_infos[b][gt_idx]["valid"]:
                    continue
                query_prob = torch.sigmoid(func_logits[b, q_idx]).detach().cpu()
                domain_label = domain_infos[b][gt_idx]["label"].detach().float().cpu()
                self.val_domain_preds.append(query_prob)
                self.val_domain_labels.append(domain_label)

            # End-to-end domain-function evaluation (v2 fix: prediction-driven).
            # Predicted domains come from NMS+merge over mask probabilities (the
            # same post-processing used for the IoU metric) and are matched to GT
            # domains by interval IoU. The objectness head is deliberately NOT
            # used as a gate: it was stuck near 0.5 in the fresh run and crushed
            # the e2e metric through spurious active queries.
            from evaluate_domain import apply_nms_merge_to_mask_logits
            seq_len_b = seq_lens[b]
            pred_labels = apply_nms_merge_to_mask_logits(
                pred_masks[b], seq_len_b, threshold=float(self.hparams.mask_threshold)
            )
            pred_ids = {int(x) for x in pred_labels if x > 0}
            pred_region = {pid: np.where(pred_labels == pid)[0] for pid in pred_ids}
            mask_prob = torch.sigmoid(
                pred_masks[b, :, :seq_len_b].float()
            ).detach().cpu().numpy()                                             # [Q, seq_len]
            func_prob = torch.sigmoid(
                func_logits[b].float()
            ).detach().cpu().numpy()                                             # [Q, C]
            pred_q = {}
            for pid, idx in pred_region.items():
                if len(idx) == 0:
                    continue
                pred_q[pid] = int(mask_prob[:, idx].mean(axis=1).argmax())
            gt_ids_b = {}
            for domain_id, d_info in enumerate(domain_infos[b], start=1):
                vis = d_info["mask"][:seq_len_b].float().detach().cpu().numpy() > 0.5
                if vis.any():
                    gt_ids_b[domain_id] = np.where(vis)[0]
            matched_pids = set()
            for gt_id, gidx in gt_ids_b.items():
                d_info = domain_infos[b][gt_id - 1]
                is_valid = bool(d_info["valid"])
                best_pid, best_iou = None, 0.0
                for pid, pidx in pred_region.items():
                    inter = float(np.intersect1d(gidx, pidx).size)
                    union = float(np.union1d(gidx, pidx).size)
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_pid, best_iou = pid, iou
                matched = best_pid is not None and best_iou >= 0.5
                if matched:
                    matched_pids.add(best_pid)
                # Validity evaluation on ALL GT domains (valid + invalid):
                # the predicted domain's validity logit vs GT function_label_valid.
                if matched:
                    self.val_e2e_validity_preds.append(
                        float(outputs["validity_logits"][b, pred_q[best_pid]].item())
                    )
                    self.val_e2e_validity_targets.append(1.0 if is_valid else 0.0)
                # e2e function metrics / 口径B only on valid GT domains.
                if is_valid:
                    self.val_e2e_gt_count += 1
                    if matched:
                        pred = func_prob[pred_q[best_pid]]
                        # 口径B: matched-domain function quality (no extra penalty)
                        self.val_e2e_matched_preds.append(torch.from_numpy(pred).float())
                        self.val_e2e_matched_labels.append(
                            d_info["label"].detach().float().cpu()
                        )
                        self.val_e2e_match_count += 1
                    else:
                        pred = np.zeros(self.hparams.num_classes)
                    self.val_e2e_domain_preds.append(torch.from_numpy(pred).float())
                    self.val_e2e_domain_labels.append(
                        d_info["label"].detach().float().cpu()
                    )
            # Unmatched predicted domains receive zero labels (penalize extras).
            self.val_e2e_extra_count += len(pred_ids - matched_pids)
            zero_label = np.zeros(self.hparams.num_classes)
            for pid in pred_ids - matched_pids:
                self.val_e2e_domain_preds.append(
                    torch.from_numpy(func_prob[pred_q[pid]]).float()
                )
                self.val_e2e_domain_labels.append(
                    torch.from_numpy(zero_label).float()
                )

        # Protein-level function evaluation (unified representation).
        protein_label = batch["protein_label"]
        self.val_protein_preds.append(torch.sigmoid(protein_logits).detach().cpu())
        self.val_protein_labels.append(protein_label)

        # Segmentation data.
        for b in range(pred_masks.size(0)):
            self.val_mask_logits.append(pred_masks[b].detach().float().cpu())
            self.val_objectness_logits.append(
                outputs["objectness_logits"][b].detach().float().cpu()
            )
            gt_labels = torch.zeros(seq_lens[b], dtype=torch.long)
            for domain_id, d_info in enumerate(domain_infos[b], start=1):
                visible = (d_info["mask"][:seq_lens[b]] > 0.5).detach().cpu()
                gt_labels[visible] = domain_id
            self.val_gt_labels.append(gt_labels)
            self.val_seq_lens.append(seq_lens[b])

    def on_validation_epoch_end(self):
        # --- Domain-level Fmax (function branch) ---
        if self.val_domain_preds:
            d_preds = torch.stack(self.val_domain_preds, dim=0)
            d_labels = torch.stack(self.val_domain_labels, dim=0)
        else:
            d_preds = torch.zeros((0, self.hparams.num_classes))
            d_labels = torch.zeros((0, self.hparams.num_classes))
        d_preds = self._gather_variable_rows(d_preds.to(self.device)).cpu()
        d_labels = self._gather_variable_rows(d_labels.to(self.device)).cpu()
        if d_preds.size(0) == 0:
            oracle_fmax = 0.0
        else:
            oracle_fmax, best_t = compute_fmax(d_preds, d_labels)
            self.log("best_threshold", best_t, sync_dist=True)
        self.log("val_oracle_domain_Fmax", oracle_fmax, prog_bar=False, sync_dist=True)
        # Compatibility alias for existing analysis scripts.
        self.log("val_Fmax", oracle_fmax, prog_bar=False, sync_dist=True)

        # --- End-to-end domain-level Fmax (penalizes misses/extras) ---
        if self.val_e2e_domain_preds:
            e_preds = torch.stack(self.val_e2e_domain_preds)
            e_labels = torch.stack(self.val_e2e_domain_labels)
        else:
            e_preds = torch.zeros((0, self.hparams.num_classes))
            e_labels = torch.zeros((0, self.hparams.num_classes))
        e_preds = self._gather_variable_rows(e_preds.to(self.device)).cpu()
        e_labels = self._gather_variable_rows(e_labels.to(self.device)).cpu()
        e2e_fmax = compute_fmax(e_preds, e_labels)[0] if e_preds.size(0) else 0.0
        self.log("val_e2e_domain_Fmax", e2e_fmax, prog_bar=True, sync_dist=True)

        # --- 口径B: matched-domain function quality (primary e2e metric) ---
        # Only the GT domains that were successfully segmented+matched count;
        # extra predicted domains are reported separately (real inference does
        # produce extras and they should be visible, not silently dropped).
        if self.val_e2e_matched_preds:
            mp = torch.stack(self.val_e2e_matched_preds)
            ml = torch.stack(self.val_e2e_matched_labels)
        else:
            mp = torch.zeros((0, self.hparams.num_classes))
            ml = torch.zeros((0, self.hparams.num_classes))
        # Every rank must enter the same collectives, including ranks with no
        # locally matched domains, otherwise four-GPU validation can deadlock.
        mp = self._gather_variable_rows(mp.to(self.device)).cpu()
        ml = self._gather_variable_rows(ml.to(self.device)).cpu()
        mfmax = compute_fmax(mp, ml)[0] if mp.size(0) else 0.0
        self.log("val_matched_domain_Fmax", mfmax, prog_bar=True, sync_dist=True)
        gt_tot = max(self.val_e2e_gt_count, 1)
        self.log("val_domain_match_rate", self.val_e2e_match_count / gt_tot,
                 prog_bar=False, sync_dist=True)
        extra_tot = self.val_e2e_match_count + self.val_e2e_extra_count
        self.log("val_domain_extra_ratio", self.val_e2e_extra_count / max(extra_tot, 1),
                 prog_bar=False, sync_dist=True)
        # Validity prediction accuracy on matched domains (does the domain have
        # an observed function label?).
        if self.val_e2e_validity_preds:
            vp = torch.tensor(self.val_e2e_validity_preds)
            vt = torch.tensor(self.val_e2e_validity_targets)
        else:
            vp = torch.zeros(0)
            vt = torch.zeros(0)
        vp_g = self._gather_variable_rows(vp.unsqueeze(1).to(self.device)).cpu().squeeze(1)
        vt_g = self._gather_variable_rows(vt.unsqueeze(1).to(self.device)).cpu().squeeze(1)
        if vp_g.numel():
            vpred = (torch.sigmoid(vp_g) > 0.5).float()
            vacc = float((vpred == vt_g).float().mean().item())
        else:
            vacc = 0.0
        self.log("val_validity_acc", vacc, prog_bar=False, sync_dist=True)

        # --- Protein-level Fmax (unified representation) ---
        pfmax = 0.0
        if self.val_protein_preds:
            p_preds = torch.cat(self.val_protein_preds, dim=0)
            p_labels = torch.cat(self.val_protein_labels, dim=0)
            p_preds = self._gather_variable_rows(p_preds.to(self.device)).cpu()
            p_labels = self._gather_variable_rows(p_labels.to(self.device)).cpu()
            pfmax, _ = compute_fmax(p_preds, p_labels)
            self.log("val_protein_Fmax", pfmax, prog_bar=True, sync_dist=True)
            p_bin = (p_preds > 0.5).int()
            pf1 = f1_score(
                p_labels.cpu().numpy(), p_bin.cpu().numpy(),
                average="micro", zero_division=0,
            )
            self.log("val_protein_f1", pf1, prog_bar=True, sync_dist=True)

        # --- Segmentation metrics (v2 fix: NMS primary, objectness-aware secondary) ---
        val_domain_iou = self._compute_domain_iou()
        self.log("val_domain_IoU", val_domain_iou, prog_bar=True, sync_dist=True)
        if self._last_domain_metrics:
            nms_m = self._last_domain_metrics["nms"]
            qa_m = self._last_domain_metrics["qa"]
            self.log("val_weighted_MCC", nms_m["weighted_mcc"], sync_dist=True)
            self.log("val_simple_MCC", nms_m["simple_mcc"], sync_dist=True)
            self.log("val_simple_IoU", nms_m["simple_iou"], sync_dist=True)
            self.log("val_IoU_queryaware", qa_m["weighted_iou"], sync_dist=True)
        # Primary selection objective: both requested tasks must be high. The
        # geometric mean strongly penalizes a checkpoint weak on either task.
        joint_score = float(np.sqrt(max(float(pfmax), 0.0) * max(val_domain_iou, 0.0)))
        self.log("val_joint_score", joint_score, prog_bar=True, sync_dist=True)
        # Three-task model-selection score. Keep val_joint_score unchanged for
        # backward comparability, while explicitly requiring the end-to-end
        # domain classification metric to be high as well.
        tri_score = float(
            (max(float(pfmax), 0.0)
             * max(val_domain_iou, 0.0)
             * max(float(e2e_fmax), 0.0)) ** (1.0 / 3.0)
        )
        self.log("val_tri_score", tri_score, prog_bar=True, sync_dist=True)

        # Cleanup
        self.val_domain_preds.clear()
        self.val_domain_labels.clear()
        self.val_e2e_domain_preds.clear()
        self.val_e2e_domain_labels.clear()
        self.val_e2e_matched_preds.clear()
        self.val_e2e_matched_labels.clear()
        self.val_e2e_validity_preds.clear()
        self.val_e2e_validity_targets.clear()
        self.val_e2e_gt_count = 0
        self.val_e2e_match_count = 0
        self.val_e2e_extra_count = 0
        self.val_protein_preds.clear()
        self.val_protein_labels.clear()
        self.val_mask_logits.clear()
        self.val_gt_labels.clear()
        self.val_objectness_logits.clear()
        self.val_seq_lens.clear()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _gather_variable_rows(self, tensor):
        """Gather [N, ...] validation tensors exactly across DDP ranks."""
        # AMP predictions may be BF16 while empty shards are constructed as
        # FP32. Collective buffers must have the same dtype on every rank.
        # All downstream metrics already convert these values to FP32.
        tensor = tensor.float().contiguous()
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        local_n = torch.tensor([tensor.size(0)], device=tensor.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_n) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_n)
        max_n = max(int(size.item()) for size in sizes)
        if max_n == 0:
            return tensor
        if tensor.size(0) < max_n:
            padding = torch.zeros(
                (max_n - tensor.size(0), *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([tensor, padding], dim=0)
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat([
            part[:int(size.item())] for part, size in zip(gathered, sizes)
        ], dim=0)

    def _compute_domain_iou(self):
        """Domain segmentation metrics (v2 fix).

        Primary metric: NMS+merge post-processing at ``mask_threshold`` (0.5),
        the same pipeline as the v3 curriculum lineage that reached IoU 0.888,
        so numbers are comparable across runs. The objectness-aware variant is
        kept as a secondary metric, plus weighted/simple MCC.
        """
        from evaluate_domain import (
            apply_query_aware_to_mask_logits,
            apply_nms_merge_to_mask_logits,
            evaluate_single_protein,
        )

        # [w_iou*w, w_mcc*w, w, s_iou, s_mcc, count]
        acc = {"nms": np.zeros(6), "qa": np.zeros(6)}
        for i in range(len(self.val_mask_logits)):
            seq_len = self.val_seq_lens[i]
            gt_labels = self.val_gt_labels[i].numpy()
            if gt_labels.max() == 0:
                continue
            weight = max(float(np.sum(gt_labels > 0)), 1.0)
            for key, fn in [
                ("nms", apply_nms_merge_to_mask_logits),
                ("qa", apply_query_aware_to_mask_logits),
            ]:
                if key == "nms":
                    pred_labels = fn(
                        self.val_mask_logits[i], seq_len,
                        threshold=float(self.hparams.mask_threshold),
                    )
                else:
                    pred_labels = fn(
                        self.val_mask_logits[i], seq_len,
                        objectness_logits=self.val_objectness_logits[i],
                        mask_threshold=self.hparams.mask_threshold,
                        objectness_threshold=self.hparams.objectness_threshold,
                    )
                metrics = evaluate_single_protein(pred_labels, gt_labels, tolerance=20)
                acc[key] += np.array([
                    metrics["iou"] * weight, metrics["mcc"] * weight, weight,
                    metrics["iou"], metrics["mcc"], 1.0,
                ])

        self._last_domain_metrics = {}
        for key, vec in acc.items():
            totals = torch.tensor(vec, device=self.device, dtype=torch.float64)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            t = totals.cpu().numpy()
            self._last_domain_metrics[key] = {
                "weighted_iou": float(t[0] / max(t[2], 1e-8)),
                "weighted_mcc": float(t[1] / max(t[2], 1e-8)),
                "simple_iou": float(t[3] / max(t[5], 1e-8)),
                "simple_mcc": float(t[4] / max(t[5], 1e-8)),
            }
        return self._last_domain_metrics["nms"]["weighted_iou"]

    def configure_optimizers(self):
        # v2 fix: per-group learning rates. Task/split heads train faster than
        # the shared encoder; a warmup phase prevents early instability.
        head_keywords = (
            "func_head", "protein_head", "fusion_mlp", "mask_embed", "mask_gate",
            "objectness_head", "query_score_head", "anchor_mlp", "dn_anchor_mlp",
            "dn_content_proj", "func_content_proj", "func_prototype_emb",
            "seg_decoder", "func_decoder",
        )
        head_params, enc_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in head_keywords):
                head_params.append(p)
            else:
                enc_params.append(p)
        lr_head = float(getattr(self.hparams, "lr_head", 3e-4))
        lr_enc = float(getattr(self.hparams, "lr_enc", self.hparams.lr))
        optimizer = torch.optim.AdamW([
            {"params": enc_params, "lr": lr_enc, "weight_decay": 0.01},
            {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
        ])

        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup_steps = int(total_steps * float(
            getattr(self.hparams, "warmup_ratio", 0.05)
        ))
        eta_min = 1e-6
        lr_ref = max(lr_head, lr_enc)

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return 0.05 + 0.95 * (step / warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            t = min(max(t, 0.0), 1.0)
            return (eta_min / lr_ref) + 0.5 * (1.0 - eta_min / lr_ref) * (
                1.0 + math.cos(math.pi * t)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_load_checkpoint(self, checkpoint):
        """Honor an explicitly lower CLI learning rate when resuming.

        Lightning normally restores the optimizer/scheduler learning rate from
        the checkpoint, which would otherwise ignore ``--lr`` on a resumed run.
        Only lower rates are allowed here so ordinary resumes remain identical.
        """
        target_lr = float(self.hparams.lr)
        saved_base_lr = float(checkpoint.get("hyper_parameters", {}).get("lr", target_lr))
        # Compare configured base rates, not the decayed per-group rates.
        # An unchanged resume must not collapse the head/encoder LR ratio.
        if target_lr >= saved_base_lr:
            return
        for optimizer_state in checkpoint.get("optimizer_states", []):
            for param_group in optimizer_state.get("param_groups", []):
                saved_lr = float(param_group.get("lr", target_lr))
                if target_lr < saved_lr:
                    param_group["lr"] = target_lr
                    param_group["initial_lr"] = target_lr

        for scheduler_state in checkpoint.get("lr_schedulers", []):
            saved_lrs = scheduler_state.get("_last_lr", [])
            if saved_lrs and target_lr < max(float(lr) for lr in saved_lrs):
                scheduler_state["base_lrs"] = [target_lr] * len(saved_lrs)
                scheduler_state["_last_lr"] = [target_lr] * len(saved_lrs)

    def _extract_gt_spans(self, domain_infos, attn_mask):
        """Extract GT domain spans from domain_infos for denoising training."""
        B = len(domain_infos)
        L = attn_mask.size(1)
        all_spans = []
        for b in range(B):
            spans = []
            for d_info in domain_infos[b]:
                mask = d_info["mask"]
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
