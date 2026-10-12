"""Read-only continuous-component audit of the selected final-region model.

The historical 1,376-protein cohort and conversion/matching/ranking functions
come from the archived reviewer check. No fitting or test-time threshold choice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/reviewer_checks_20260914"))
from v7_joint_common9015 import connected_components, collapse_predictions  # noqa: E402
from quasf.decoders import Prediction  # noqa: E402
from quasf.metrics import segmentation  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def truth(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = len(row["sequence"])
    masks, labels, valid = [], [], []
    for domain in row["domains"]:
        mask = np.zeros(length, bool)
        for start, end in domain["segments"]:  # source is one-based inclusive
            mask[max(0, start - 1):min(length, end)] = True
        if not mask.any():
            continue
        label = np.zeros(256, bool)
        label[domain.get("function_cluster_l2_indices", [])] = True
        masks.append(mask)
        labels.append(label)
        valid.append(bool(domain.get("function_label_valid", False)))
    return (np.asarray(masks, bool).reshape(-1, length),
            np.asarray(labels, bool).reshape(-1, 256), np.asarray(valid, bool))


def rank(scores: np.ndarray, targets: np.ndarray, budget: int) -> dict:
    score = scores.ravel()
    target = targets.ravel()
    order = np.argsort(-score, kind="stable")
    score, target = score[order], target[order]
    starts = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1]
    ends = np.r_[starts[1:], len(score)]
    group_tp = np.add.reduceat(target.astype(np.int64), starts)
    cumulative = np.r_[0, np.cumsum(group_tp)]

    def at(k: int) -> dict:
        k = min(k, len(score))
        index = min(np.searchsorted(ends, k, side="left"), len(ends) - 1)
        correct = float(cumulative[index] +
                        (k - starts[index]) * group_tp[index] / (ends[index] - starts[index]))
        return {"calls": k, "expected_correct": correct,
                "cutoff_score": float(score[starts[index]]),
                "cutoff_tie_size": int(ends[index] - starts[index])}

    return {"at_9149": at(budget),
            "curve": [at(int(k)) for k in np.unique(np.linspace(1, min(len(score), 20000), 401, dtype=int))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.27)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort = {r["accession"] for r in csv.DictReader(args.cohort.open(newline=""))}
    rows = {r["uniprot_acc"]: r for r in map(json.loads, args.source.open(encoding="utf-8"))
            if r["uniprot_acc"] in cohort}
    assert len(cohort) == len(rows) == 1376, (len(cohort), len(rows))
    scores, targets = [], []
    recovered, references, reference_labels = 0, 0, 0
    call_count, correct_count = 0, 0
    selected = set()
    per_protein = []
    for line in args.predictions.open(encoding="utf-8"):
        output = json.loads(line)
        accession = output["accession"]
        if accession not in cohort:
            continue
        selected.add(accession)
        source = rows[accession]
        length = len(source["sequence"])
        masks, labels, valid = truth(source)
        component_masks, component_labels, component_valid, _ = connected_components(
            masks, labels, valid, length)
        references += len(component_masks)
        reference_labels += int(component_labels[component_valid].sum())
        predictions = []
        for position, item in enumerate(output["predictions"]):
            mask = np.zeros(length, bool)
            for start, end in item["segments"]:  # archived zero-based, end-exclusive
                mask[max(0, start):min(length, end)] = True
            if not mask.any():
                continue
            predictions.append(Prediction(int(item["source_query_id"]), mask,
                                          np.asarray(item["function_probs"], np.float32),
                                          float(item["mask_score"]), float(item["objectness"])))
        collapsed = collapse_predictions(predictions, length, Prediction)
        row, pairs = segmentation(collapsed, component_masks, length, thresholds=(0.5,))
        recovered += row["domain_tp@0.5"]
        matched = {i: j for i, j, _ in pairs[0.5]}
        protein_scores, protein_targets = [], []
        for index, prediction in enumerate(collapsed):
            gt = matched.get(index)
            if gt is not None and not component_valid[gt]:
                continue
            score = prediction.function_probs
            target = component_labels[gt] if gt is not None else np.zeros(256, bool)
            scores.append(score)
            targets.append(target)
            protein_scores.append(score)
            protein_targets.append(target)
        per_protein.append({"accession": accession, "reference_components": len(component_masks),
                            "predicted_components": len(collapsed), "matched_components": row["domain_tp@0.5"],
                            "eligible_prediction_regions": len(protein_scores),
                            "observed_reference_labels": int(component_labels[component_valid].sum())})
    assert len(selected) == len(cohort) == 1376, (len(selected), len(cohort))
    assert references == 4303 and reference_labels == 5606, (references, reference_labels)
    score = np.asarray(scores, np.float32).reshape(-1, 256)
    target = np.asarray(targets, bool).reshape(-1, 256)
    ranked = rank(score, target, 9149)
    fixed_calls = score >= np.float32(args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fixed_budget_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ranked["curve"][0].keys())
        writer.writeheader()
        writer.writerows(ranked["curve"])
    with (args.output / "per_protein.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_protein[0].keys())
        writer.writeheader()
        writer.writerows(per_protein)
    summary = {"cohort": len(cohort), "reference_components": references,
               "reference_region_labels": reference_labels,
               "recovered_components_iou05": recovered,
               "eligible_prediction_regions": len(score),
               "candidate_calls": score.size,
               "fixed_budget_9149": ranked["at_9149"],
               "validation_fixed_threshold": args.threshold,
               "threshold_calls": int(fixed_calls.sum()),
               "threshold_correct": int((fixed_calls & target).sum()),
               "source_sha256": digest(args.source),
               "predictions_sha256": digest(args.predictions),
               "cohort_sha256": digest(args.cohort),
               "script_sha256": digest(Path(__file__))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
