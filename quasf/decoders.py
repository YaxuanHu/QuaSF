"""Pure NumPy decoders. Legacy A/B mirror the inspected v6 implementation."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from .common import runs


@dataclass
class Prediction:
    query_id: int
    mask: np.ndarray
    function_probs: np.ndarray
    mask_score: float
    objectness: float


def overlap(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def interval_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = a[1] - a[0] + b[1] - b[0] - inter
    return inter / union if union else 0.0


def cover(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / (b[1] - b[0])


def legacy_decode_labels(probs, threshold):
    intervals = []
    for row in probs:
        intervals.extend([[s, e, float(row[s:e].mean())] for s, e in runs(row > threshold)])
    # Stable order is deliberate: mirrors the original Python sorted / NMS.
    pending = sorted(intervals, key=lambda x: x[2], reverse=True)
    kept = []
    while pending:
        current = pending.pop(0)
        kept.append(current)
        # Original implementation checks length only for remaining intervals.
        pending = [p for p in pending if not (
            interval_iou(current, p) >= 0.5 or cover(current, p) >= 0.8 or p[1] - p[0] < 10)]
    for _ in range(3):
        if not kept:
            break
        ordered = sorted(kept, key=lambda p: p[0])
        merged = [ordered[0]]
        for current in ordered[1:]:
            last = merged[-1]
            iou, c1, c2 = interval_iou(last, current), cover(last, current), cover(current, last)
            strong = iou >= 0.3 or c1 >= 0.7 or c2 >= 0.7
            weak = iou >= 0.15 or c1 >= 0.4 or c2 >= 0.4
            start, end = min(last[0], current[0]), max(last[1], current[1])
            ratio = min(last[2], current[2]) / max(last[2], current[2])
            if (strong or (weak and current[0] - last[1] <= 5)) and ratio >= 0.6 and end - start <= 300:
                l1, l2 = last[1] - last[0], current[1] - current[0]
                merged[-1] = [start, end, (last[2] * l1 + current[2] * l2) / (l1 + l2)]
            else:
                merged.append(current)
        if len(merged) == len(kept):
            break
        kept = merged
    labels = np.zeros(probs.shape[1], dtype=np.int32)
    for i, (start, end, _) in enumerate(kept, 1):
        labels[start:end] = i
    return labels


def decode(mask_logits, func_logits, obj_logits, config):
    probs = expit(np.asarray(mask_logits, dtype=np.float32))
    funcs = expit(np.asarray(func_logits, dtype=np.float32))
    objects = expit(np.asarray(obj_logits, dtype=np.float32))
    mode = config["mode"]
    threshold = config.get("mask_threshold", 0.3)
    if mode in ("legacy_nms", "legacy_gap"):
        labels = legacy_decode_labels(probs, threshold)
        masks = [labels == i for i in np.unique(labels) if i > 0]
        if mode == "legacy_gap":
            spans = sorted((int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1]) + 1) for m in masks)
            merged = []
            for s, e in spans:
                if merged and s - merged[-1][1] < config.get("merge_gap", 30):
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            masks = []
            for s, e in merged:
                mask = np.zeros(probs.shape[1], bool)
                mask[s:e] = True
                masks.append(mask)
        answer = []
        for mask in masks:
            q = int(probs[:, mask].mean(axis=1).argmax())
            answer.append(Prediction(q, mask, funcs[q], float(probs[q, mask].mean()), float(objects[q])))
        return answer
    if mode not in ("query", "query_objectness"):
        raise ValueError(f"Unknown decoder: {mode}")
    minimum = config.get("min_residues", 10)
    binary = probs > threshold
    candidates = []
    for q, mask in enumerate(binary):
        if mask.sum() < minimum:
            continue
        score = float(probs[q, mask].mean())
        if mode == "query_objectness":
            if objects[q] < config.get("objectness_threshold", 0.5):
                continue
            score *= float(objects[q])
        candidates.append((q, score))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    kept = []
    for q, _ in candidates:
        # Only mask IoU suppression: no cross-query interval merging / containment removal.
        if not any(overlap(binary[q], binary[k]) >= config.get("nms_iou", 0.5) for k in kept):
            kept.append(q)
    # Residue ownership uses mask probability in BOTH C and D. Ties use query ID.
    kept.sort()
    while kept:
        active = np.where(binary[kept], probs[kept], -np.inf)
        owners = active.argmax(axis=0)
        masks = [(owners == i) & binary[q] for i, q in enumerate(kept)]
        valid = [q for q, mask in zip(kept, masks) if mask.sum() >= minimum]
        if valid == kept:
            return [Prediction(q, mask, funcs[q], float(probs[q, mask].mean()), float(objects[q]))
                    for q, mask in zip(kept, masks)]
        # Removed tiny domains relinquish their residues; deterministic reassignment.
        kept = valid
    return []
