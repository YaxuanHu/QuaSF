"""M2 broadcast and connected-component adapters used in paper Table 2."""
import csv
import gzip
import json
import numpy as np

def parse_region(text):
    start, end = text.split("-")
    return int(start), int(end)

def load_regions(path):
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["accession"]] = [
                [parse_region(segment) for segment in domain]
                for domain in json.loads(row["predicted_domains_json"])
            ]
    return result

def load_m2_probabilities(path):
    result = {}
    with gzip.open(path, "rt") as handle:
        for row in map(json.loads, handle):
            queries = [np.asarray(q["function_probs"], np.float32) for q in row["raw_queries"]]
            result[row["accession"]] = np.max(queries, axis=0) if queries else np.zeros(256, np.float32)
    return result

def external_predictions(domains, length, probabilities, Prediction):
    answer = []
    for domain_id, domain in enumerate(domains):
        mask = np.zeros(length, dtype=bool)
        for start, end in domain:  # source files use one-based inclusive coordinates
            mask[max(0, start - 1):min(length, end)] = True
        if mask.any():
            answer.append(Prediction(domain_id, mask, probabilities, 1.0, 1.0))
    return answer

def collapse_predictions(predictions, length, Prediction):
    """Mirror v7: binarize the predicted union and recover contiguous components."""
    union = np.logical_or.reduce([p.mask for p in predictions]) if predictions else np.zeros(length, bool)
    answer = []
    active = False
    start = 0
    for i, value in enumerate(np.r_[union, False]):
        if value and not active:
            start, active = i, True
        elif active and not value:
            mask = np.zeros(length, bool)
            mask[start:i] = True
            # Retain the identity of the source query contributing most residues.
            source = max(predictions, key=lambda p: (int((p.mask & mask).sum()), p.mask_score, -p.query_id))
            answer.append(Prediction(source.query_id, mask, source.function_probs,
                                     source.mask_score, source.objectness))
            active = False
    return answer

def connected_components(gt_masks, gt_labels, gt_valid, length):
    union = np.logical_or.reduce(gt_masks, axis=0) if len(gt_masks) else np.zeros(length, bool)
    components = []
    active = False
    start = 0
    for i, value in enumerate(np.r_[union, False]):
        if value and not active:
            start, active = i, True
        elif active and not value:
            mask = np.zeros(length, bool)
            mask[start:i] = True
            contributors = np.asarray([(mask & source).any() for source in gt_masks], bool)
            labels = np.asarray(gt_labels[contributors], bool).any(axis=0).astype(np.float32)
            valid = bool(np.asarray(gt_valid)[contributors].any())
            components.append((mask, labels, valid, int(contributors.sum())))
            active = False
    if not components:
        return (np.zeros((0, length), bool), np.zeros((0, gt_labels.shape[1]), np.float32),
                np.zeros(0, bool), [])
    return (np.stack([x[0] for x in components]), np.stack([x[1] for x in components]),
            np.asarray([x[2] for x in components], bool), [x[3] for x in components])
