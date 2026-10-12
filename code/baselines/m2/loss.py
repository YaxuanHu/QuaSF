"""
Loss functions for end-to-end domain segmentation + protein-level classification.

Strictly aligned with hemenet/query_v2/loss.py.
Task loss: all query logits max-pooled → protein-level focal loss.
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
    def match(self, pred_masks, gt_masks):
        """
        pred_masks: [Q, L] logits
        gt_masks: List[Tensor[L]]
        """
        Q = pred_masks.size(0)
        K = len(gt_masks)
        if K == 0:
            return []

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

    def forward(self, pred_masks, gt_masks, matches):
        """
        pred_masks: [Q, L]
        gt_masks: List[Tensor[L]]
        matches: List[(q_idx, gt_idx)]
        """
        if len(matches) == 0:
            return torch.tensor(0.0, device=pred_masks.device)

        loss = 0.0
        for q_idx, gt_idx in matches:
            pred = pred_masks[q_idx]
            gt = gt_masks[gt_idx]
            loss += F.binary_cross_entropy_with_logits(pred, gt)
            loss += self.dice(pred.sigmoid().unsqueeze(0), gt.unsqueeze(0)).squeeze()

        return loss / len(matches)


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
                bce = F.binary_cross_entropy_with_logits(pred, gt)
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
        score_probs = torch.sigmoid(query_scores)
        valid = residue_mask.float()

        loss = F.binary_cross_entropy(
            score_probs * valid,
            domain_mask * valid,
            reduction='none',
        )
        loss = (loss * valid).sum() / valid.sum().clamp(min=1)
        return loss


class ProteinClassificationLoss(nn.Module):
    """
    Focal loss on max-pooled query logits → protein-level multi-label classification.
    This is the original v7 approach: all queries pooled to protein level.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, class_logits, targets):
        """
        class_logits: [B, Q, C]
        targets: [B, C] protein-level multi-label
        """
        # Max-pool over queries → protein-level logits
        seq_logits, _ = class_logits.max(dim=1)  # [B, C]
        prob = torch.sigmoid(seq_logits)
        ce = F.binary_cross_entropy_with_logits(seq_logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce * ((1 - p_t) ** self.gamma)
        return loss.mean()
