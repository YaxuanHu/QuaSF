"""Continuous-localization protocol used in paper Table 1."""
import csv
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

def parse_region(text):
    start, end = text.split("-")
    return int(start), int(end)

def load_csv(path):
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["accession"]] = [[parse_region(x) for x in d]
                                         for d in json.loads(row["predicted_domains_json"])]
    return result

def labels_from_domains(length, domains):
    labels = np.zeros(length, dtype=np.int32)
    for domain_id, domain in enumerate(domains, 1):
        for start, end in domain:
            labels[max(0, start - 1):min(length, end)] = domain_id
    return labels

def collapse_truth_to_components(labels):
    binary = labels > 0
    result = np.zeros(len(labels), dtype=np.int32)
    domain_id = 0
    active = False
    for i, value in enumerate(binary):
        if value and not active:
            domain_id += 1
        active = bool(value)
        if active:
            result[i] = domain_id
    return result

def domain_iou(pred, truth):
    pred_ids = np.unique(pred[pred > 0])
    gt_ids = np.unique(truth[truth > 0])
    if not len(gt_ids):
        return float(not len(pred_ids))
    if not len(pred_ids):
        return 0.0
    matrix = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pid in enumerate(pred_ids):
        pmask = pred == pid
        for j, gid in enumerate(gt_ids):
            gmask = truth == gid
            union = np.logical_or(pmask, gmask).sum()
            matrix[i, j] = np.logical_and(pmask, gmask).sum() / union if union else 0.0
    rows, cols = linear_sum_assignment(-matrix)
    lengths = np.asarray([(truth == gid).sum() for gid in gt_ids], float)
    values = np.zeros(len(gt_ids))
    values[cols] = matrix[rows, cols]
    return float(np.average(values, weights=lengths))

def boundary_mcc(pred, truth, tolerance=20):
    pb = set((np.flatnonzero(pred[1:] != pred[:-1]) + 1).tolist())
    gb = set((np.flatnonzero(truth[1:] != truth[:-1]) + 1).tolist())
    tp = 0
    matched = set()
    for p in pb:
        candidates = [(abs(p - g), g) for g in gb if g not in matched and abs(p - g) <= tolerance]
        if candidates:
            _, g = min(candidates)
            matched.add(g)
            tp += 1
    fp, fn = len(pb) - tp, len(gb) - tp
    tn = len(truth) - tp - fp - fn
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / den) if den else 0.0
