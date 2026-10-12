"""Prepare reference-region mean/max features for query pretraining."""
from pathlib import Path
import argparse
import json
import torch

def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def domain_indices(segments, length):
    parts = [torch.arange(max(0, int(a)), min(length, int(b))) for a, b in segments]
    parts = [p for p in parts if p.numel()]
    return torch.cat(parts).unique() if parts else torch.empty(0, dtype=torch.long)

def pooled_feature(residue_emb, indices):
    selected = residue_emb.index_select(0, indices)
    return torch.cat((selected.float().mean(0), selected.float().amax(0)))

def build_cache(root: Path, split_file: Path, output: Path, shard_size=2048, max_records=None):
    output.mkdir(parents=True, exist_ok=True)
    xs, ys, meta, shard = [], [], [], 0

    def flush():
        nonlocal xs, ys, meta, shard
        if not xs:
            return
        name = f"shard_{shard:05d}.pt"
        torch.save({"x": torch.stack(xs).half(), "y": torch.stack(ys).to(torch.uint8), "meta": meta}, output / name)
        print(f"saved {name}: {len(xs)} domains", flush=True)
        xs, ys, meta, shard = [], [], [], shard + 1

    missing = invalid = 0
    for record_index, record in enumerate(iter_jsonl(split_file)):
        if max_records is not None and record_index >= max_records:
            break
        feature_path = root / record["esm_path"]
        if not feature_path.exists():
            missing += 1
            continue
        saved = torch.load(feature_path, map_location="cpu")
        emb = saved["residue_emb"]
        length = min(len(record["sequence"]), emb.shape[0])
        for domain in record["domains"]:
            if not domain.get("function_label_valid", False):
                continue
            indices = domain_indices(domain["segments"], length)
            labels = domain.get("function_cluster_l2_indices", [])
            if not indices.numel() or not labels:
                invalid += 1
                continue
            y = torch.zeros(256, dtype=torch.uint8)
            y[torch.as_tensor(labels, dtype=torch.long)] = 1
            xs.append(pooled_feature(emb, indices))
            ys.append(y)
            meta.append({"accession": record["uniprot_acc"], "domain_id": domain["domain_id"]})
            if len(xs) >= shard_size:
                flush()
    flush()
    manifest = {"split": str(split_file), "shards": shard, "missing_features": missing,
                "invalid_domains": invalid, "feature": "concat(mean,max)", "classes": 256}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)

def main() -> None:
    """Build one split of reference-region pretraining shards."""
    p = argparse.ArgumentParser()
    for name in ("root", "split", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard-size", type=int, default=2048)
    a = p.parse_args()
    build_cache(a.root, a.split, a.output, a.shard_size)

if __name__ == "__main__":
    main()
