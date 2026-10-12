"""Add the independent Seq30 test split to end_to_end_domain_function_v1."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="data/dataset_v2/end_to_end_domain_function_v1",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / args.output_dir

    source = root / "data/dataset_v2/anchor_mil_oracle_v1/test.jsonl"
    features = root / "data/dataset_v2/esm_features_test_v2"
    core = root / "data/dataset_v2/core/domain_go_core_v1.jsonl"
    mapping_path = (
        root / "data/dataset_v2/domain_go_pseudo_v1/go_to_cluster.json"
    )
    mapping = json.loads(mapping_path.read_text())["go_to_cluster"]
    vocab = json.loads((output / "cluster_vocab.json").read_text())
    l2_index = vocab["level_2"]["cluster_to_index"]
    l1_index = vocab["level_1"]["cluster_to_index"]

    observed = defaultdict(dict)
    for row in read_jsonl(core):
        cluster = mapping.get(row["go_id"])
        if cluster is None:
            continue
        observed[(row["uniprot_acc"], row["domain_id"])][row["go_id"]] = {
            "go_id": row["go_id"],
            "go_name": row.get("go_name", row["go_id"]),
            "level_2_cluster_id": cluster["level_2_cluster_id"],
            "level_1_cluster_id": cluster["level_1_cluster_id"],
            "confidence_tier": row.get("confidence_tier"),
            "mapping_path": row.get("mapping_path"),
            "support_count": row.get("support_count"),
        }

    stats = Counter()
    frequency = Counter()
    excluded = []
    test_path = output / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as writer:
        for row in read_jsonl(source):
            acc = row["uniprot_acc"]
            sequence = row["sequence"]
            length = len(sequence)
            feature_path = features / f"{acc}.pt"
            invalid = []
            for domain in row["domains"]:
                for start, end in domain["segments"]:
                    if not 1 <= start <= end <= length:
                        invalid.append(
                            f"{domain['domain_id']}:{start}-{end}/{length}"
                        )
            if invalid or not feature_path.is_file():
                excluded.append(
                    {
                        "uniprot_acc": acc,
                        "reason": (
                            "invalid_coordinates" if invalid else "missing_esm"
                        ),
                        "details": invalid,
                        "strict_primary_test": bool(
                            row.get("strict_primary_test", False)
                        ),
                    }
                )
                continue

            domains = []
            stats["proteins"] += 1
            stats["strict_primary_test_proteins"] += bool(
                row.get("strict_primary_test", False)
            )
            for domain in row["domains"]:
                stats["domains"] += 1
                labels = sorted(
                    observed.get((acc, domain["domain_id"]), {}).values(),
                    key=lambda value: value["go_id"],
                )
                l2_ids = sorted(
                    {label["level_2_cluster_id"] for label in labels}
                )
                l1_ids = sorted(
                    {label["level_1_cluster_id"] for label in labels}
                )
                if labels:
                    stats["domains_with_observed_function"] += 1
                    stats["observed_domain_go_labels"] += len(labels)
                    stats["observed_domain_l2_labels"] += len(l2_ids)
                    frequency.update(l2_ids)
                    status = "observed"
                else:
                    stats["domains_with_unknown_function"] += 1
                    status = "unknown"
                domains.append(
                    {
                        "domain_id": domain["domain_id"],
                        "segments": domain["segments"],
                        "ted_consensus": domain.get("ted_consensus"),
                        "function_label_status": status,
                        "function_label_valid": bool(labels),
                        "function_cluster_l2_ids": l2_ids,
                        "function_cluster_l2_indices": [
                            l2_index[value] for value in l2_ids
                        ],
                        "function_cluster_l1_ids": l1_ids,
                        "function_cluster_l1_indices": [
                            l1_index[value] for value in l1_ids
                        ],
                        "source_go_labels": labels,
                        "unobserved_functions_are_unknown": True,
                    }
                )
            writer.write(
                json.dumps(
                    {
                        "format": "dataset-v2-end-to-end-domain-function-v1",
                        "uniprot_acc": acc,
                        "split": "test",
                        "sequence": sequence,
                        "protein_length": length,
                        "sequence_sha256": hashlib.sha256(
                            sequence.encode("ascii")
                        ).hexdigest(),
                        "esm_path": str(feature_path.relative_to(root)),
                        "esm_format": "dataset-v2-esm-residue-v1",
                        "strict_primary_test": bool(
                            row.get("strict_primary_test", False)
                        ),
                        "domains": domains,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    excluded_path = output / "test_excluded.jsonl"
    with excluded_path.open("w", encoding="utf-8") as writer:
        for row in excluded:
            writer.write(json.dumps(row) + "\n")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["test"] = {
        **dict(stats),
        "level_2_cluster_frequency": dict(sorted(frequency.items())),
    }
    manifest["sources"]["test_protein_domains"] = str(source.relative_to(root))
    manifest["sources"]["test_esm_features"] = str(features.relative_to(root))
    manifest["test_policy"] = {
        "split": "independent Seq30 test from anchor_mil_oracle_v1",
        "strict_primary_test": "retained as a per-protein boolean",
        "validation": "ESM present and all coordinates within frozen sequence",
        "excluded_proteins": len(excluded),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    checksum_path = output / "SHA256SUMS.json"
    checksums = json.loads(checksum_path.read_text())
    for path in (test_path, excluded_path, manifest_path):
        checksums[path.name] = sha256(path)
    checksum_path.write_text(json.dumps(checksums, indent=2) + "\n")
    print(json.dumps({"test": dict(stats), "excluded": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
