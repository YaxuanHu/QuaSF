"""
Domain segmentation evaluation utilities.
NMS + iterative merge on predicted domain intervals.
"""

import numpy as np
import torch


def mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union else 0.0


def apply_query_aware_to_mask_logits(
    mask_logits,
    valid_len,
    objectness_logits=None,
    mask_threshold=0.5,
    objectness_threshold=0.5,
    mask_nms_iou=0.6,
    min_residues=10,
):
    """Convert query masks to domain IDs without splitting discontinuous masks.

    One retained query is one domain. Multiple separated positive segments from
    that query therefore receive the same domain ID. Overlapping residues are
    assigned to the retained query with the highest mask probability.
    """
    probs = torch.sigmoid(mask_logits[:, :valid_len].float()).detach().cpu().numpy()
    if objectness_logits is None:
        objectness = probs.mean(axis=1)
    else:
        objectness = torch.sigmoid(objectness_logits.float()).detach().cpu().numpy()

    candidates = []
    for q in range(probs.shape[0]):
        binary = probs[q] >= mask_threshold
        if int(binary.sum()) < min_residues:
            continue
        if float(objectness[q]) < objectness_threshold:
            continue
        score = float(objectness[q]) * float(probs[q][binary].mean())
        candidates.append((q, binary, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    kept = []
    for candidate in candidates:
        if any(mask_iou(candidate[1], other[1]) >= mask_nms_iou
               for other in kept):
            continue
        kept.append(candidate)

    labels = np.zeros(valid_len, dtype=int)
    if not kept:
        return labels
    kept_probs = np.stack([probs[q] for q, _, _ in kept], axis=0)
    winner = kept_probs.argmax(axis=0)
    for idx, (_, binary, _) in enumerate(kept):
        owned = np.logical_and(binary, winner == idx)
        labels[owned] = idx + 1
    return labels


# ======================
# 1D NMS + Merge utilities
# ======================

def iou_1d(a, b):
    """a, b: [start, end, score]"""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def cover_ratio(a, b):
    """覆盖率：b 中有多少被 a 覆盖"""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    len_b = b[1] - b[0]
    return inter / len_b if len_b > 0 else 0.0


def nms_1d_strict(preds, iou_thresh=0.5, cover_thresh=0.8, min_len=10):
    """1D NMS: 按 score 降序贪心保留。"""
    if not preds:
        return []
    preds = sorted(preds, key=lambda x: x[2], reverse=True)
    keep = []
    while preds:
        cur = preds.pop(0)
        keep.append(cur)
        remaining = []
        for p in preds:
            iou = iou_1d(cur, p)
            cover = cover_ratio(cur, p)
            length = p[1] - p[0]
            if iou >= iou_thresh or cover >= cover_thresh or length < min_len:
                continue
            remaining.append(p)
        preds = remaining
    return keep


def merge_intervals_strict(preds, max_gap=5, max_domain_len=300):
    """相邻区间合并。"""
    if not preds:
        return []
    preds = sorted(preds, key=lambda x: x[0])
    merged = [preds[0]]
    for cur in preds[1:]:
        last = merged[-1]
        iou = iou_1d(last, cur)
        cover1 = cover_ratio(last, cur)
        cover2 = cover_ratio(cur, last)
        gap = cur[0] - last[1]
        strong_overlap = iou >= 0.3 or cover1 >= 0.7 or cover2 >= 0.7
        weak_overlap = iou >= 0.15 or cover1 >= 0.4 or cover2 >= 0.4
        close_enough = gap <= max_gap
        score_ratio = min(last[2], cur[2]) / max(last[2], cur[2])
        new_start = min(last[0], cur[0])
        new_end = max(last[1], cur[1])
        if (
            (strong_overlap or (weak_overlap and close_enough))
            and score_ratio >= 0.6
            and (new_end - new_start) <= max_domain_len
        ):
            len1 = last[1] - last[0]
            len2 = cur[1] - cur[0]
            new_score = (last[2] * len1 + cur[2] * len2) / (len1 + len2)
            merged[-1] = [new_start, new_end, new_score]
        else:
            merged.append(cur)
    return merged


def iterative_merge(preds, rounds=3):
    """多轮 merge，直到稳定"""
    for _ in range(rounds):
        new_preds = merge_intervals_strict(preds)
        if len(new_preds) == len(preds):
            break
        preds = new_preds
    return preds


def extract_intervals_from_mask(probs_q, threshold=0.5):
    """从单个 query 的 sigmoid 概率中提取连续区间。"""
    binary = (probs_q > threshold).astype(int)
    intervals = []
    L = len(binary)
    i = 0
    while i < L:
        if binary[i]:
            start = i
            while i < L and binary[i]:
                i += 1
            end = i
            score = float(probs_q[start:end].mean())
            intervals.append([start, end, score])
        else:
            i += 1
    return intervals


def apply_nms_merge_to_mask_logits(mask_logits, valid_len, threshold=0.5):
    """
    对 mask_logits 做 NMS + iterative merge，再转回 per-residue domain labels。

    Args:
        mask_logits: Tensor [Q, L] raw logits
        valid_len: int, actual sequence length
        threshold: sigmoid threshold

    Returns:
        np.array [valid_len] with domain IDs (0=linker/bg)
    """
    probs = torch.sigmoid(mask_logits[:, :valid_len].float()).cpu().numpy()  # [Q, valid_len]
    Q, L = probs.shape

    # 1) 从所有 query 提取区间
    all_intervals = []
    for q in range(Q):
        intervals = extract_intervals_from_mask(probs[q], threshold)
        all_intervals.extend(intervals)

    if not all_intervals:
        return np.zeros(L, dtype=int)

    # 2) NMS 去重
    intervals_nms = nms_1d_strict(all_intervals, iou_thresh=0.5, cover_thresh=0.8, min_len=10)

    # 3) 多轮 merge
    intervals_merged = iterative_merge(intervals_nms, rounds=3)

    # 4) 转回 per-residue labels
    labels = np.zeros(L, dtype=int)
    for domain_id, (s, e, _score) in enumerate(intervals_merged, start=1):
        s, e = int(s), int(e)
        labels[s:e] = domain_id

    return labels


def evaluate_single_protein(pred_labels, gt_labels, tolerance=20):
    """
    Evaluate domain segmentation for a single protein.

    Args:
        pred_labels: np.array [L] with domain IDs (0=bg)
        gt_labels: np.array [L] with domain IDs (0=bg)
        tolerance: int, boundary tolerance in residues

    Returns:
        dict with 'iou' and 'mcc' keys
    """
    # IoU: residue-level overlap
    pred_binary = (pred_labels > 0).astype(int)
    gt_binary = (gt_labels > 0).astype(int)

    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum((pred_binary + gt_binary) > 0)
    iou = intersection / union if union > 0 else 0.0

    # MCC: boundary prediction
    # Extract boundaries from labels
    pred_bounds = _extract_boundaries(pred_labels)
    gt_bounds = _extract_boundaries(gt_labels)

    L = len(pred_labels)
    tp = 0
    for pb in pred_bounds:
        for gb in gt_bounds:
            if abs(pb - gb) <= tolerance:
                tp += 1
                break

    fp = len(pred_bounds) - tp
    fn = sum(1 for gb in gt_bounds if not any(abs(pb - gb) <= tolerance for pb in pred_bounds))
    tn = L - tp - fp - fn

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {"iou": iou, "mcc": mcc}


def _extract_boundaries(labels):
    """Extract boundary positions from domain labels."""
    boundaries = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.append(i)
    return boundaries
