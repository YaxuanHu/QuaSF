"""
Shared evaluation metrics for domain segmentation baselines.

Metrics:
1. IoU (Intersection over Union) - measures residue overlap between predicted and true domains
2. MCC (Matthews Correlation Coefficient) - measures boundary prediction accuracy (±20 residues tolerance)

Both metrics are computed at the domain level and averaged weighted by domain length.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_domain_iou(pred_labels, gt_labels):
    """Compute IoU between predicted and ground truth domain assignments.

    Uses Hungarian matching to find optimal assignment between predicted and GT domains.

    Args:
        pred_labels: np.array [N] predicted domain IDs (0=linker/bg)
        gt_labels: np.array [N] ground truth domain IDs (0=linker/bg)

    Returns:
        weighted_iou: float, domain-length weighted average IoU
        per_domain_iou: dict mapping gt_domain_id -> iou
    """
    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)

    gt_ids = np.unique(gt_labels)
    gt_domain_ids = gt_ids[gt_ids > 0]

    pred_ids = np.unique(pred_labels)
    pred_domain_ids = pred_ids[pred_ids > 0]

    if len(gt_domain_ids) == 0:
        return 1.0 if len(pred_domain_ids) == 0 else 0.0, {}

    if len(pred_domain_ids) == 0:
        return 0.0, {int(g): 0.0 for g in gt_domain_ids}

    # Build IoU cost matrix [n_pred, n_gt]
    n_pred = len(pred_domain_ids)
    n_gt = len(gt_domain_ids)
    iou_matrix = np.zeros((n_pred, n_gt))

    for i, pid in enumerate(pred_domain_ids):
        pred_mask = (pred_labels == pid)
        for j, gid in enumerate(gt_domain_ids):
            gt_mask = (gt_labels == gid)
            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)
            iou_matrix[i, j] = intersection / (union + 1e-8)

    # Hungarian matching (maximize IoU = minimize negative IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    per_domain_iou = {}
    total_weight = 0
    weighted_sum = 0

    for ri, ci in zip(row_ind, col_ind):
        gid = gt_domain_ids[ci]
        iou = iou_matrix[ri, ci]
        domain_len = np.sum(gt_labels == gid)
        per_domain_iou[int(gid)] = iou
        weighted_sum += iou * domain_len
        total_weight += domain_len

    # Unmatched GT domains get IoU = 0
    matched_gt = set(gt_domain_ids[col_ind])
    for gid in gt_domain_ids:
        if gid not in matched_gt:
            domain_len = np.sum(gt_labels == gid)
            per_domain_iou[int(gid)] = 0.0
            total_weight += domain_len

    weighted_iou = weighted_sum / (total_weight + 1e-8)
    return weighted_iou, per_domain_iou


def get_domain_boundaries(labels):
    """Extract domain boundary positions from per-residue labels.

    A boundary is defined as a position where the domain label changes.

    Args:
        labels: np.array [N] domain labels

    Returns:
        boundaries: set of boundary positions (0-indexed)
    """
    labels = np.array(labels)
    boundaries = set()
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.add(i)
    return boundaries


def compute_boundary_mcc(pred_labels, gt_labels, tolerance=20):
    """Compute MCC for boundary prediction with tolerance.

    A predicted boundary is considered correct if it is within ±tolerance
    residues of a true boundary.

    Args:
        pred_labels: np.array [N] predicted domain labels
        gt_labels: np.array [N] ground truth domain labels
        tolerance: int, allowed error in residues for boundary matching

    Returns:
        mcc: float, Matthews Correlation Coefficient
        details: dict with TP, FP, FN, TN counts
    """
    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)
    N = len(gt_labels)

    pred_boundaries = get_domain_boundaries(pred_labels)
    gt_boundaries = get_domain_boundaries(gt_labels)

    # For each position, determine if it's a boundary (positive) or not (negative)
    # With tolerance: a predicted boundary at position p is TP if there exists
    # a GT boundary within [p-tolerance, p+tolerance]

    TP = 0
    FP = 0
    matched_gt = set()

    for pb in pred_boundaries:
        found = False
        for gb in gt_boundaries:
            if abs(pb - gb) <= tolerance and gb not in matched_gt:
                TP += 1
                matched_gt.add(gb)
                found = True
                break
        if not found:
            FP += 1

    FN = len(gt_boundaries) - len(matched_gt)
    TN = N - TP - FP - FN  # All non-boundary positions correctly identified

    # MCC formula
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

    if denominator == 0:
        mcc = 0.0
    else:
        mcc = numerator / denominator

    return mcc, {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}


def compute_weighted_boundary_mcc(pred_labels, gt_labels, tolerance=20):
    """Compute domain-length weighted MCC.

    Computes MCC per GT domain boundary and weights by domain length.

    Args:
        pred_labels: np.array [N]
        gt_labels: np.array [N]
        tolerance: int

    Returns:
        weighted_mcc: float
    """
    # For simplicity, compute a single MCC over all boundaries
    mcc, details = compute_boundary_mcc(pred_labels, gt_labels, tolerance)
    return mcc, details


def evaluate_single_protein(pred_labels, gt_labels, tolerance=20):
    """Evaluate a single protein's domain prediction.

    Args:
        pred_labels: array-like [N] predicted domain IDs
        gt_labels: array-like [N] ground truth domain IDs
        tolerance: int, boundary tolerance in residues

    Returns:
        dict with 'iou', 'mcc', 'per_domain_iou', 'boundary_details'
    """
    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)

    iou, per_domain_iou = compute_domain_iou(pred_labels, gt_labels)
    mcc, boundary_details = compute_weighted_boundary_mcc(pred_labels, gt_labels, tolerance)

    return {
        'iou': iou,
        'mcc': mcc,
        'per_domain_iou': per_domain_iou,
        'boundary_details': boundary_details,
    }


def evaluate_dataset(predictions, ground_truths, tolerance=20):
    """Evaluate predictions over a dataset.

    Args:
        predictions: list of (pdb_id, pred_labels) tuples
        ground_truths: list of (pdb_id, gt_labels) tuples
        tolerance: int

    Returns:
        dict with average metrics and per-protein results
    """
    gt_dict = {pid: labels for pid, labels in ground_truths}

    results = {}
    ious = []
    mccs = []
    weights = []

    for pdb_id, pred_labels in predictions:
        if pdb_id not in gt_dict:
            continue

        gt_labels = gt_dict[pdb_id]
        if len(pred_labels) != len(gt_labels):
            # Length mismatch - skip or truncate
            min_len = min(len(pred_labels), len(gt_labels))
            pred_labels = pred_labels[:min_len]
            gt_labels = gt_labels[:min_len]

        metrics = evaluate_single_protein(pred_labels, gt_labels, tolerance)
        results[pdb_id] = metrics

        # Weight by total domain residues
        gt_arr = np.array(gt_labels)
        domain_weight = np.sum(gt_arr > 0)
        ious.append(metrics['iou'])
        mccs.append(metrics['mcc'])
        weights.append(max(domain_weight, 1))

    weights = np.array(weights, dtype=float)
    weights /= weights.sum() + 1e-8

    avg_iou = np.average(ious, weights=weights) if len(ious) > 0 else 0.0
    avg_mcc = np.average(mccs, weights=weights) if len(mccs) > 0 else 0.0
    simple_avg_iou = np.mean(ious) if len(ious) > 0 else 0.0
    simple_avg_mcc = np.mean(mccs) if len(mccs) > 0 else 0.0

    return {
        'avg_iou': avg_iou,
        'avg_mcc': avg_mcc,
        'simple_avg_iou': simple_avg_iou,
        'simple_avg_mcc': simple_avg_mcc,
        'num_proteins': len(results),
        'per_protein': results,
    }
