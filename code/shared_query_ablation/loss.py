"""
Loss functions for end-to-end domain segmentation + domain-level classification.

Key difference from protein-level version:
Task loss is computed per-query per-domain after Hungarian matching.
Only domains with valid function labels contribute to classification loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        pred: [N, L] sigmoid probabilities
        target: [N, L] binary
        """
        inter = (pred * target).sum(-1)
        union = pred.sum(-1) + target.sum(-1)
        return 1 - (2 * inter + self.eps) / (union + self.eps)


class DomainHungarianMatcher:
    def __init__(self):
        self.dice = DiceLoss()

    @torch.no_grad()
    def match(self, pred_masks, gt_masks, valid_mask=None):
        """
        pred_masks: [Q, L] logits
        gt_masks: List[Tensor[L]]
        """
        Q = pred_masks.size(0)
        K = len(gt_masks)
        if K == 0:
            return []

        if valid_mask is not None:
            pred_masks = pred_masks[:, valid_mask]
            gt_masks = [m[valid_mask] for m in gt_masks]
        pred_prob = pred_masks.sigmoid()
        cost = torch.zeros(Q, K, device=pred_masks.device)
        for k in range(K):
            cost[:, k] = self.dice(pred_prob, gt_masks[k].unsqueeze(0).expand(Q, -1))

        q_ind, k_ind = linear_sum_assignment(cost.cpu().numpy())
        return list(zip(q_ind.tolist(), k_ind.tolist()))


class MultiDomainMaskLoss(nn.Module):
    """BCE + Dice loss on Hungarian-matched query-GT pairs."""
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()

    def forward(self, pred_masks, gt_masks, matches, valid_mask=None,
                unmatched_weight=0.0):
        """
        pred_masks: [Q, L]
        gt_masks: List[Tensor[L]]
        matches: List[(q_idx, gt_idx)]
        """
        if valid_mask is not None:
            pred_masks = pred_masks[:, valid_mask]
            gt_masks = [m[valid_mask] for m in gt_masks]

        loss = pred_masks.sum() * 0.0
        for q_idx, gt_idx in matches:
            pred = pred_masks[q_idx]
            gt = gt_masks[gt_idx]
            # Balanced BCE: average in-domain and out-of-domain pixels separately
            # so the (usually 5-10x larger) background does not collapse the
            # predicted mask probability (observed collapse: p50 ~ 0.03 while
            # mask_loss kept decreasing). Dice term keeps precision-recall balance.
            bce_all = F.binary_cross_entropy_with_logits(
                pred, gt, reduction="none"
            )
            pos = gt > 0.5
            neg = ~pos
            terms = []
            if pos.any():
                terms.append(bce_all[pos].mean())
            if neg.any():
                terms.append(bce_all[neg].mean())
            bce = torch.stack(terms).mean() if terms else bce_all.mean()
            loss += bce
            loss += self.dice(pred.sigmoid().unsqueeze(0), gt.unsqueeze(0)).squeeze()

        if matches:
            loss = loss / len(matches)

        # Explicitly suppress masks from unmatched queries. Without this term,
        # unused queries are free to emit extra domains at inference time.
        if unmatched_weight > 0:
            matched_queries = {q for q, _ in matches}
            unmatched = [q for q in range(pred_masks.size(0))
                         if q not in matched_queries]
            if unmatched:
                background = torch.zeros_like(pred_masks[unmatched])
                no_object_loss = F.binary_cross_entropy_with_logits(
                    pred_masks[unmatched], background
                )
                loss = loss + unmatched_weight * no_object_loss
        return loss


class DenoisingMaskLoss(nn.Module):
    """
    Direct supervision loss for denoising queries.
    No Hungarian matching needed - each DN query has a known GT target.
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()

    def forward(self, dn_mask_logits, dn_gt_masks):
        """
        Args:
            dn_mask_logits: [B, K, L] raw logits from DN queries
            dn_gt_masks: List[List[Tensor[L]]] per-sample GT masks for each DN query
        """
        B, K, L = dn_mask_logits.shape
        total_loss = 0.0
        count = 0

        for b in range(B):
            for k in range(K):
                gt = dn_gt_masks[b][k]  # [L]
                if gt.sum() == 0:
                    continue
                pred = dn_mask_logits[b, k]  # [L]
                # Balanced BCE (in/out domain separately) to prevent background
                # pixels from collapsing the predicted mask probability.
                bce_all = F.binary_cross_entropy_with_logits(
                    pred, gt, reduction="none"
                )
                pos = gt > 0.5
                neg = ~pos
                terms = []
                if pos.any():
                    terms.append(bce_all[pos].mean())
                if neg.any():
                    terms.append(bce_all[neg].mean())
                bce = torch.stack(terms).mean() if terms else bce_all.mean()
                dice = self.dice(pred.sigmoid().unsqueeze(0), gt.unsqueeze(0)).squeeze()
                total_loss += bce + dice
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=dn_mask_logits.device)
        return total_loss / count


class DomainSelectionLoss(nn.Module):
    """
    Auxiliary loss to encourage query_score_head to assign high scores
    to positions within domain regions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, query_scores, domain_mask, residue_mask):
        """
        Args:
            query_scores: [B, L] raw scores from scoring head
            domain_mask: [B, L] binary GT domain mask (1=domain, 0=linker/pad)
            residue_mask: [B, L] attention mask (1=valid, 0=pad)
        """
        valid = residue_mask.float()
        # Logit-space BCE is numerically stable and safe under BF16/FP16 AMP.
        loss = F.binary_cross_entropy_with_logits(
            query_scores, domain_mask.float(), reduction='none'
        )
        loss = (loss * valid).sum() / valid.sum().clamp(min=1)
        return loss


class DomainClassificationLoss(nn.Module):
    """
    Per-domain focal loss on Hungarian-matched query logits.
    
    Key difference from ProteinClassificationLoss:
    - Instead of max-pooling all queries → protein-level target,
      each matched query gets its own domain-level target.
    - Only domains with function_label_valid=True contribute to loss.
    - Unmatched queries or unknown-function domains are skipped.
    """
    def __init__(self, alpha=0.75, gamma=2.0, negative_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.negative_weight = negative_weight

    def forward(self, class_logits, matches_list, domain_infos, device):
        """
        Args:
            class_logits: [B, Q, C] per-query classification logits
            matches_list: List[List[(q_idx, gt_idx)]] per-sample Hungarian matches
            domain_infos: List[List[dict]] per-sample domain info with 'label' and 'valid'
            device: torch device

        Returns:
            scalar loss (averaged over valid matched pairs)
        """
        B, Q, C = class_logits.shape
        total_loss = 0.0
        count = 0

        for b in range(B):
            matches = matches_list[b]
            d_infos = domain_infos[b]

            for q_idx, gt_idx in matches:
                # Only compute loss for domains with valid function labels
                if not d_infos[gt_idx]["valid"]:
                    continue

                # Get this query's logits and this domain's target
                query_logits = class_logits[b, q_idx]  # [C]
                domain_target = d_infos[gt_idx]["label"].to(device)  # [C]

                # Focal loss
                prob = torch.sigmoid(query_logits)
                ce = F.binary_cross_entropy_with_logits(
                    query_logits, domain_target, reduction='none'
                )
                p_t = prob * domain_target + (1 - prob) * (1 - domain_target)
                alpha_t = (
                    self.alpha * domain_target
                    + (1.0 - self.alpha) * (1.0 - domain_target)
                )
                focal = alpha_t * ce * ((1 - p_t) ** self.gamma)

                # Function labels are extremely sparse. Averaging over all 256
                # classes makes the positive signal nearly disappear. Normalize
                # positives and negatives separately so each labeled domain has
                # a meaningful classification gradient.
                pos = domain_target > 0.5
                neg = ~pos
                pos_loss = focal[pos].mean() if pos.any() else focal.sum() * 0.0
                neg_loss = focal[neg].mean() if neg.any() else focal.sum() * 0.0
                total_loss += pos_loss + self.negative_weight * neg_loss
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / count


class ProteinRepresentationLoss(nn.Module):
    """
    Focal loss on the protein-level prediction (max-pooled over query logits).

    Matches the loss recipe of the previous e2e_protein_cls model
    (QuerySelectionMaskDinoV2) that reached protein Fmax 0.907: protein logits
    are the per-class max over queries, supervised by the union of domain
    labels with focal loss (gamma=2) to combat the sparse long-tailed labels.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, protein_logits, protein_label):
        """
        Args:
            protein_logits: [B, C] max-pooled protein-level logits
            protein_label:  [B, C] protein-level multi-label target
        Returns:
            scalar focal loss
        """
        prob = torch.sigmoid(protein_logits)
        ce = F.binary_cross_entropy_with_logits(
            protein_logits, protein_label, reduction="none"
        )
        p_t = prob * protein_label + (1 - prob) * (1 - protein_label)
        loss = ce * ((1 - p_t) ** self.gamma)
        return loss.mean()


class GtConditionedFunctionLoss(nn.Module):
    """
    Direct supervision for ground-truth-conditioned function queries.

    Each GT-conditioned function query pools the memory over an exact GT domain
    mask and is supervised by that domain's function label directly (no
    Hungarian matching), denoising-style. Samples drawn from a protein with no
    valid-function GT domains carry an all-zero label and are skipped.
    """
    def __init__(self, alpha=0.75, gamma=2.0, negative_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.negative_weight = negative_weight

    def forward(self, gt_func_logits, gt_func_labels):
        """
        Args:
            gt_func_logits: [B, K, C] GT-conditioned function logits
            gt_func_labels:  [B, K, C] GT function labels (zero for empty samples)
        Returns:
            scalar BCE loss over non-empty entries
        """
        valid = gt_func_labels.sum(dim=-1) > 0  # [B, K]
        if not valid.any():
            return torch.tensor(
                0.0, device=gt_func_logits.device, requires_grad=True
            )
        logits = gt_func_logits[valid]
        targets = gt_func_labels[valid]
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * ce * ((1 - p_t) ** self.gamma)

        # Function labels are sparse. Give each valid GT domain one balanced
        # positive/negative contribution instead of averaging over all classes.
        pos = targets > 0.5
        neg = ~pos
        pos_sum = (focal * pos).sum(dim=-1)
        pos_count = pos.sum(dim=-1).clamp(min=1)
        neg_sum = (focal * neg).sum(dim=-1)
        neg_count = neg.sum(dim=-1).clamp(min=1)
        return (pos_sum / pos_count + self.negative_weight * neg_sum / neg_count).mean()
