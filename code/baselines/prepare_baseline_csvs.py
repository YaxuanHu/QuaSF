#!/usr/bin/env python3
"""Convert records.jsonl into baseline-ready CSVs and a filtered records JSONL.

The baseline evaluate.py scripts expect a CSV with columns:
    id, uniprot, sequence, domain_regions

and a structures_dir containing PDB files named {id}.pdb or {uniprot}.pdb.
Our AFDB structures are named 'AF-{uniprot}-F1-model_v6.pdb', so we set
id to the structure filename stem (without .pdb) so find_pdb() works.

domain_regions format:
    <domain1_seg1_start>-<end>,<domain1_seg2_start>-<end>;
    <domain2_seg1_start>-<end>;
    ...

Reads split directly from records.jsonl and writes:
    data/baseline_csvs/{train,val,test}.csv
    data/baseline_csvs/records.jsonl   (filtered records with structures + domains)

The filtered records.jsonl is the recommended downstream input for the MAPE-PPI
segmenter: it contains the same schema as data/dataset_v1/records.jsonl but only
for proteins that have both a PDB structure and at least one TED domain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def segments_to_region_string(ted_domains: list[dict]) -> str:
    """Convert TED domains (with segments) to baseline domain_regions string."""
    if not ted_domains:
        return ""

    domain_strings = []
    for domain in ted_domains:
        segments = domain.get("segments", [])
        if not segments:
            continue
        seg_strs = [f"{int(start)}-{int(end)}" for start, end in segments]
        domain_strings.append(",".join(seg_strs))

    return ";".join(domain_strings)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/dataset_v1/records.jsonl"),
        help="Path to records.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/baseline_csvs"),
        help="Output directory for CSVs and filtered records.jsonl",
    )
    parser.add_argument(
        "--records-jsonl-out",
        type=Path,
        default=None,
        help="Optional path for the filtered records.jsonl (default: OUT_DIR/records.jsonl)",
    )
    args = parser.parse_args()

    records_out = args.records_jsonl_out or args.out_dir / "records.jsonl"

    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    filtered_records: list[dict] = []
    skipped_no_domain = 0
    skipped_no_structure = 0
    total = 0

    with args.records.open() as handle:
        for line in handle:
            rec = json.loads(line)
            total += 1

            sequence = rec.get("sequence", "")
            ted_domains = rec.get("ted_domains", [])
            split = rec.get("split", "")
            uniprot = rec.get("uniprot_acc", "")
            structure_file = rec.get("structure_file", "")

            if not sequence or len(sequence) < 10:
                continue
            if not ted_domains:
                skipped_no_domain += 1
                continue
            if not structure_file:
                skipped_no_structure += 1
                continue
            if split not in rows_by_split:
                continue

            domain_regions = segments_to_region_string(ted_domains)
            if not domain_regions:
                skipped_no_domain += 1
                continue

            rows_by_split[split].append({
                "id": Path(structure_file).stem,
                "uniprot": uniprot,
                "sequence": sequence,
                "domain_regions": domain_regions,
            })
            filtered_records.append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        out_path = args.out_dir / f"{split}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Wrote {split}.csv: {len(rows)} proteins -> {out_path}")

    with records_out.open("w") as handle:
        for rec in filtered_records:
            handle.write(json.dumps(rec) + "\n")
    print(f"Wrote records.jsonl: {len(filtered_records)} proteins -> {records_out}")

    print(
        f"\nTotal records scanned: {total}"
        f"\n  Skipped (no domain): {skipped_no_domain}"
        f"\n  Skipped (no structure file): {skipped_no_structure}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
