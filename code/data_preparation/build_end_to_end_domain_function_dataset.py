"""Build end-to-end protein domain segmentation/function manifests."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


FORMAT = "dataset-v2-end-to-end-domain-function-v1"


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


def build(project_root, output_dir):
    root = Path(project_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    (output / "examples").mkdir(exist_ok=True)

    domain_dir = root / "data/dataset_v2/anchor_mil_oracle_v2"
    core_path = root / "data/dataset_v2/core/domain_go_core_v1.jsonl"
    map_path = root / "data/dataset_v2/domain_go_pseudo_v1/go_to_cluster.json"
    map_obj = json.loads(map_path.read_text(encoding="utf-8"))
    go_map = map_obj["go_to_cluster"]

    l2_ids = sorted({value["level_2_cluster_id"] for value in go_map.values()})
    l1_ids = sorted({value["level_1_cluster_id"] for value in go_map.values()})
    l2_index = {cluster_id: i for i, cluster_id in enumerate(l2_ids)}
    l1_index = {cluster_id: i for i, cluster_id in enumerate(l1_ids)}
    vocab = {
        "format": "dataset-v2-function-cluster-vocab-v1",
        "source_label_space_version": map_obj.get("label_space_version"),
        "primary_level": 2,
        "level_2": {
            "count": len(l2_ids),
            "cluster_ids": l2_ids,
            "cluster_to_index": l2_index,
        },
        "level_1": {
            "count": len(l1_ids),
            "cluster_ids": l1_ids,
            "cluster_to_index": l1_index,
        },
    }
    vocab_path = output / "cluster_vocab.json"
    vocab_path.write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Stable join key: (protein accession, TED domain ID).
    observed = defaultdict(dict)
    unmapped = Counter()
    for row in read_jsonl(core_path):
        go_id = row["go_id"]
        cluster = go_map.get(go_id)
        if cluster is None:
            unmapped[go_id] += 1
            continue
        key = (row["uniprot_acc"], row["domain_id"])
        observed[key][go_id] = {
            "go_id": go_id,
            "go_name": row.get("go_name", go_id),
            "level_2_cluster_id": cluster["level_2_cluster_id"],
            "level_1_cluster_id": cluster["level_1_cluster_id"],
            "confidence_tier": row.get("confidence_tier"),
            "mapping_path": row.get("mapping_path"),
            "support_count": row.get("support_count"),
        }

    split_summaries = {}
    outputs = []
    example_written = False
    for split in ("train", "val"):
        destination = output / f"{split}.jsonl"
        outputs.append(destination)
        stats = Counter()
        cluster_frequency = Counter()
        with destination.open("w", encoding="utf-8") as writer:
            for row in read_jsonl(domain_dir / f"{split}.jsonl"):
                stats["proteins"] += 1
                domains = []
                for domain in row["domains"]:
                    stats["domains"] += 1
                    labels = sorted(
                        observed.get(
                            (row["uniprot_acc"], domain["domain_id"]), {}
                        ).values(),
                        key=lambda value: value["go_id"],
                    )
                    domain_l2 = sorted(
                        {label["level_2_cluster_id"] for label in labels}
                    )
                    domain_l1 = sorted(
                        {label["level_1_cluster_id"] for label in labels}
                    )
                    if labels:
                        status = "observed"
                        stats["domains_with_observed_function"] += 1
                        stats["observed_domain_go_labels"] += len(labels)
                        stats["observed_domain_l2_labels"] += len(domain_l2)
                        cluster_frequency.update(domain_l2)
                    else:
                        status = "unknown"
                        stats["domains_with_unknown_function"] += 1
                    domains.append(
                        {
                            "domain_id": domain["domain_id"],
                            "segments": domain["segments"],
                            "ted_consensus": domain.get("ted_consensus"),
                            "function_label_status": status,
                            "function_label_valid": bool(labels),
                            "function_cluster_l2_ids": domain_l2,
                            "function_cluster_l2_indices": [
                                l2_index[value] for value in domain_l2
                            ],
                            "function_cluster_l1_ids": domain_l1,
                            "function_cluster_l1_indices": [
                                l1_index[value] for value in domain_l1
                            ],
                            "source_go_labels": labels,
                            "unobserved_functions_are_unknown": True,
                        }
                    )

                output_row = {
                    "format": FORMAT,
                    "uniprot_acc": row["uniprot_acc"],
                    "split": split,
                    "sequence": row["sequence"],
                    "protein_length": row["protein_length"],
                    "sequence_sha256": row["sequence_sha256"],
                    "esm_path": row["esm_path"],
                    "esm_format": row.get("esm_format"),
                    "domains": domains,
                }
                writer.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                if not example_written and any(
                    domain["function_label_valid"] for domain in domains
                ):
                    (output / "examples" / f"{row['uniprot_acc']}.json").write_text(
                        json.dumps(output_row, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    example_written = True
        split_summaries[split] = {
            **dict(stats),
            "level_2_cluster_frequency": dict(sorted(cluster_frequency.items())),
        }

    manifest = {
        "format": FORMAT,
        "splits": split_summaries,
        "label_policy": {
            "primary_target": "multi-label Level-2 function clusters",
            "observed_source": "core/domain_go_core_v1.jsonl",
            "missing_domain_function": "unknown; mask classification loss",
            "absence_is_negative": False,
            "pseudo_labels_included": False,
        },
        "coordinate_policy": "1-based inclusive; discontinuous segments allowed",
        "cluster_counts": {"level_2": len(l2_ids), "level_1": len(l1_ids)},
        "unmapped_observed_go_rows": sum(unmapped.values()),
        "sources": {
            "protein_domains": str(domain_dir.relative_to(root)),
            "domain_go": str(core_path.relative_to(root)),
            "go_to_cluster": str(map_path.relative_to(root)),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checksums = {
        path.name: sha256(path)
        for path in [vocab_path, *outputs, manifest_path]
    }
    (output / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="data/dataset_v2/end_to_end_domain_function_v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
