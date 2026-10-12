"""Unified v1 metrics. Historical scores are intentionally not reused."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import runs
from .decoders import overlap


PROTOCOL = {
    "version": "quasf_decode_unified_v1",
    "coordinates": "zero-based half-open segments; complete boolean domain masks",
    "iou": "max-total-IoU Hungarian; unmatched GT zero; GT-length weights within protein",
    "detection": "max-cardinality matching above threshold, then max total IoU",
    "boundary": "unique internal endpoints of all domain mask runs; max-cardinality tolerance matching, then min distance",
    "rf": "threshold-valid one-to-one detection matches; unknown matched domains ignored; missed observed GT forced silent; all invalid/unmatched predictions extras",
    "fmax": "primary macro-all-instances P/R then F; zero-label row recall=0; zero-call precision=0; additionally micro; thresholds 0..1 step .01",
    "empty_boundary": "P/R/F1=1 if both sets empty; MCC=0 if denominator zero",
    "empty_domains": "coverage and instance IoU=1 if both sets empty; instance IoU=0 if extras but no GT",
    "weights": "cross-protein weighted scores use GT union residue count; no GT => weight zero",
}


def iou_matrix(predictions, gt_masks):
    return np.asarray([[overlap(p.mask, g) for g in gt_masks] for p in predictions], float).reshape(len(predictions), len(gt_masks))


def matches(matrix, threshold=None):
    if not matrix.size:
        return []
    if threshold is None:
        score = matrix
    else:
        valid = matrix >= threshold
        score = valid * (min(matrix.shape) + 1 + matrix)
    rows, cols = linear_sum_assignment(-score)
    return [(int(i), int(j), float(matrix[i, j])) for i, j in zip(rows, cols)
            if threshold is None or matrix[i, j] >= threshold]


def boundaries(masks, length):
    return sorted({int(x) for mask in masks for run in runs(mask) for x in run if 0 < x < length})


def boundary_scores(pred_masks, gt_masks, length, tolerance=20):
    pred, gt = boundaries(pred_masks, length), boundaries(gt_masks, length)
    tp = 0
    if pred and gt:
        distance = np.abs(np.subtract.outer(pred, gt))
        quality = 1 - distance / (tolerance + 1)
        utility = (distance <= tolerance) * (min(len(pred), len(gt)) + 1 + quality)
        rows, cols = linear_sum_assignment(-utility)
        tp = int(np.sum(distance[rows, cols] <= tolerance))
    fp, fn = len(pred) - tp, len(gt) - tp
    tn = max(length - 1 - tp - fp - fn, 0)
    den = float(np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    precision = tp / len(pred) if pred else float(not gt)
    recall = tp / len(gt) if gt else float(not pred)
    return {"boundary_tp": tp, "boundary_fp": fp, "boundary_fn": fn,
            "boundary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "boundary_precision": precision, "boundary_recall": recall,
            "boundary_mcc": (tp * tn - fp * fn) / den if den else 0.}


def segmentation(predictions, gt_masks, length, thresholds=(0.5, 0.75), tolerance=20):
    pred_union = np.logical_or.reduce([p.mask for p in predictions]) if predictions else np.zeros(length, bool)
    gt_union = np.logical_or.reduce(gt_masks) if len(gt_masks) else np.zeros(length, bool)
    matrix = iou_matrix(predictions, gt_masks)
    best = matches(matrix)
    values = np.zeros(len(gt_masks))
    for _, j, value in best:
        values[j] = value
    weights = np.asarray([g.sum() for g in gt_masks], float)
    n_pred, n_gt = len(predictions), len(gt_masks)
    result = {
        "coverage_iou": overlap(pred_union, gt_union) if (pred_union | gt_union).any() else 1.,
        "domain_iou": float(np.average(values, weights=weights)) if weights.sum() else float(n_pred == 0),
        "domain_iou_unweighted": float(values.mean()) if n_gt else float(n_pred == 0),
        "gt_residues": int(gt_union.sum()), "pred_count": n_pred, "gt_count": n_gt,
        "count_abs_error": abs(n_pred - n_gt), "count_exact": int(n_pred == n_gt),
        "oversegmented": int(n_pred > n_gt), "undersegmented": int(n_pred < n_gt),
    }
    result.update(boundary_scores([p.mask for p in predictions], gt_masks, length, tolerance))
    detection_matches = {}
    for tau in thresholds:
        detection_matches[tau] = matches(matrix, tau)
        tp = len(detection_matches[tau])
        result.update({f"domain_tp@{tau}": tp, f"domain_fp@{tau}": n_pred - tp, f"domain_fn@{tau}": n_gt - tp})
    return result, detection_matches


class Fmax:
    """Streaming macro-all-output and micro F; missed GT cannot emit at t=0."""
    def __init__(self):
        self.thresholds = np.linspace(0, 1, 101)
        self.n = 0
        self.p = np.zeros(101)
        self.r = np.zeros(101)
        self.tp = np.zeros(101)
        self.fp = np.zeros(101)
        self.fn = np.zeros(101)

    def add(self, probabilities, label, silent=False):
        probabilities, label = np.asarray(probabilities), np.asarray(label, bool)
        predicted = np.zeros((101, len(label)), bool) if silent else probabilities[None, :] >= self.thresholds[:, None]
        tp = (predicted & label).sum(axis=1)
        count = predicted.sum(axis=1)
        positives = label.sum()
        self.p += np.divide(tp, count, out=np.zeros(101), where=count > 0)
        self.r += tp / positives if positives else np.zeros(101)
        self.tp += tp
        self.fp += count - tp
        self.fn += positives - tp
        self.n += 1

    def result(self):
        if not self.n:
            return {"n": 0, "macro_fmax": None, "micro_fmax": None}
        p, r = self.p / self.n, self.r / self.n
        f = np.divide(2 * p * r, p + r, out=np.zeros(101), where=p + r > 0)
        micro = np.divide(2 * self.tp, 2 * self.tp + self.fp + self.fn,
                          out=np.zeros(101), where=2 * self.tp + self.fp + self.fn > 0)
        i, j = int(f.argmax()), int(micro.argmax())
        return {"n": self.n, "macro_fmax": float(f[i]), "macro_threshold": float(self.thresholds[i]),
                "macro_precision": float(p[i]), "macro_recall": float(r[i]),
                "micro_fmax": float(micro[j]), "micro_threshold": float(self.thresholds[j])}


def add_rf(strict, conditional, predictions, gt_labels, gt_valid, pairs):
    matched_pred, matched_gt = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
    for i, j, _ in pairs:
        if gt_valid[j]:
            strict.add(predictions[i].function_probs, gt_labels[j])
            conditional.add(predictions[i].function_probs, gt_labels[j])
    for j, label in enumerate(gt_labels):
        if gt_valid[j] and j not in matched_gt:
            strict.add(np.zeros_like(label), label, silent=True)
    for i, prediction in enumerate(predictions):
        if i not in matched_pred:
            strict.add(prediction.function_probs, np.zeros_like(prediction.function_probs))
    return {"observed_missed": sum(bool(v) and j not in matched_gt for j, v in enumerate(gt_valid)),
            "extra_predictions": len(predictions) - len(matched_pred),
            "unknown_matched_ignored": sum(not bool(gt_valid[j]) for _, j, _ in pairs)}
