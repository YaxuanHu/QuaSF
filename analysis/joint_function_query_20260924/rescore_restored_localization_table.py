"""Score the paper's 9,015-protein continuous-localization cohort."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reviewer_checks_20260914"))
from v7_protocol_common9015 import labels_from_domains, collapse_truth_to_components, domain_iou, boundary_mcc, load_csv

def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("predictions", "source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline", action="append", default=[], metavar="NAME=CSV")
    a = p.parse_args()
    source = {r["uniprot_acc"]: r for r in map(json.loads, a.source.open(encoding="utf-8"))}
    predictions = list(map(json.loads, a.predictions.open(encoding="utf-8")))
    cohort = [r for r in predictions if r["accession"] not in {"P06868", "Q58969"}]
    if len(cohort) != 9015 or len({r["accession"] for r in cohort}) != 9015:
        raise ValueError("Table 1 requires the complete, unique 9015-protein cohort")
    baselines = {name: load_csv(Path(path)) for name, path in (s.split("=", 1) for s in a.baseline)}
    rows = []
    for row in cohort:
        accession = row["accession"]
        record = source[accession]
        length = len(record["sequence"])
        truth = collapse_truth_to_components(labels_from_domains(length, [d["segments"] for d in record["domains"]]))
        methods = {"QuaSF": [[[start+1, end] for start, end in d["segments"]] for d in row["predictions"]]}
        methods.update({name: values.get(accession, []) for name, values in baselines.items()})
        for name, domains in methods.items():
            predicted = labels_from_domains(length, domains)
            rows.append(dict(method=name, accession=accession, weight=max(int((truth > 0).sum()), 1), iou=domain_iou(predicted, truth), mcc=boundary_mcc(predicted, truth)))
    result = {}
    for name in ["QuaSF", *baselines]:
        selected = [r for r in rows if r["method"] == name]
        weights = np.array([r["weight"] for r in selected])
        iou = np.array([r["iou"] for r in selected]); mcc = np.array([r["mcc"] for r in selected])
        result[name] = dict(cohort=9015, weighted_iou=float(np.average(iou, weights=weights)), mean_iou=float(iou.mean()), weighted_mcc=float(np.average(mcc, weights=weights)), mean_mcc=float(mcc.mean()))
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "per_protein.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
