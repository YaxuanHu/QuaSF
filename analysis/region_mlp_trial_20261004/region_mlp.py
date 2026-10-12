"""Train one shared region MLP and apply it to fixed segmenter outputs.

The classifier is trained only on reference domains from the training split.
Chainsaw and Merizo share the same checkpoint and inference code.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def load_cache(path: Path):
    shards = sorted(path.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no cache shards under {path}")
    values = [torch.load(p, map_location="cpu") for p in shards]
    return torch.cat([x["x"] for x in values]).float(), torch.cat([x["y"] for x in values]).float()


class RegionMLP(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, output_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


def macro_fmax(prob, labels):
    thresholds = torch.linspace(0, 1, 101)
    best = (-1.0, 0.0, 0.0, 0.0)
    labels = labels.bool()
    for threshold in thresholds:
        pred = prob >= threshold
        tp = (pred & labels).sum(1).float()
        precision = torch.where(pred.sum(1) > 0, tp / pred.sum(1), torch.zeros_like(tp)).mean()
        recall = torch.where(labels.sum(1) > 0, tp / labels.sum(1), torch.zeros_like(tp)).mean()
        score = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.tensor(0.0))
        if score.item() > best[0]:
            best = (score.item(), threshold.item(), precision.item(), recall.item())
    return {"macro_fmax": best[0], "threshold": best[1], "precision": best[2], "recall": best[3]}


@torch.no_grad()
def validate(model, x, y, device, batch_size=2048):
    model.eval()
    probs = []
    for (batch,) in DataLoader(TensorDataset(x), batch_size=batch_size):
        probs.append(torch.sigmoid(model(batch.to(device))).cpu())
    return macro_fmax(torch.cat(probs), y)


def train(train_cache: Path, val_cache: Path, output: Path, device="cuda", epochs=30,
          batch_size=1024, lr=3e-4, seed=42, patience=6):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_x, train_y = load_cache(train_cache)
    val_x, val_y = load_cache(val_cache)
    positives = train_y.sum(0)
    pos_weight = ((len(train_y) - positives) / positives.clamp_min(1)).clamp(1, 50).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True,
                        generator=generator, num_workers=2, pin_memory=True)
    model = RegionMLP(input_dim=train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_score, stale = -1.0, 0
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step()
            total += loss.item() * len(bx)
        metrics = validate(model, val_x, val_y, device)
        row = {"epoch": epoch, "loss": total / len(train_x), **metrics}
        history.append(row); print(json.dumps(row), flush=True)
        if metrics["macro_fmax"] > best_score + 1e-5:
            best_score, stale = metrics["macro_fmax"], 0
            torch.save({"state_dict": model.state_dict(), "input_dim": train_x.shape[1],
                        "hidden_dim": 512, "classes": 256, "feature": "concat(mean,max)",
                        "val": metrics, "epoch": epoch, "seed": seed}, output)
        else:
            stale += 1
            if stale >= patience:
                break
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")


def parse_domains(value):
    answer = []
    for domain in json.loads(value):
        answer.append([[int(x) for x in segment.split("-")] for segment in domain])
    return answer


@torch.no_grad()
def predict(root: Path, split_file: Path, regions_csv: Path, checkpoint: Path, output: Path,
            device="cuda", max_records=None):
    records = {r["uniprot_acc"]: r for r in iter_jsonl(split_file)}
    regions = {r["accession"]: parse_domains(r["predicted_domains_json"])
               for r in csv.DictReader(regions_csv.open(newline=""))}
    saved_model = torch.load(checkpoint, map_location="cpu")
    model = RegionMLP(saved_model["input_dim"], saved_model["hidden_dim"], saved_model["classes"]).to(device)
    model.load_state_dict(saved_model["state_dict"]); model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if output.suffix == ".gz" else open
    with opener(output, "wt", encoding="utf-8") as handle:
        for record_index, (accession, domains) in enumerate(regions.items()):
            if max_records is not None and record_index >= max_records:
                break
            record = records[accession]
            saved = torch.load(root / record["esm_path"], map_location="cpu")
            emb = saved["residue_emb"]
            length = min(len(record["sequence"]), emb.shape[0])
            features, kept = [], []
            for domain in domains:
                # Segmenter files use one-based inclusive coordinates.
                zero_based = [[max(0, a - 1), min(length, b)] for a, b in domain]
                indices = domain_indices(zero_based, length)
                if indices.numel():
                    features.append(pooled_feature(emb, indices)); kept.append(domain)
            probs = torch.sigmoid(model(torch.stack(features).to(device))).cpu().numpy() if features else []
            row = {"accession": accession, "domains": [
                {"segments": d, "function_probs": p.tolist()} for d, p in zip(kept, probs)]}
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"saved {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-cache")
    build.add_argument("--root", type=Path, required=True); build.add_argument("--split", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True); build.add_argument("--shard-size", type=int, default=2048)
    build.add_argument("--max-records", type=int)
    fit = sub.add_parser("train")
    fit.add_argument("--train-cache", type=Path, required=True); fit.add_argument("--val-cache", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True); fit.add_argument("--device", default="cuda")
    fit.add_argument("--epochs", type=int, default=30); fit.add_argument("--batch-size", type=int, default=1024)
    fit.add_argument("--lr", type=float, default=3e-4); fit.add_argument("--seed", type=int, default=42)
    infer = sub.add_parser("predict")
    infer.add_argument("--root", type=Path, required=True); infer.add_argument("--split", type=Path, required=True)
    infer.add_argument("--regions", type=Path, required=True); infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True); infer.add_argument("--device", default="cuda")
    infer.add_argument("--max-records", type=int)
    args = parser.parse_args()
    if args.command == "build-cache": build_cache(args.root, args.split, args.output, args.shard_size, args.max_records)
    elif args.command == "train": train(args.train_cache, args.val_cache, args.output, args.device,
                                           args.epochs, args.batch_size, args.lr, args.seed)
    else: predict(args.root, args.split, args.regions, args.checkpoint, args.output, args.device, args.max_records)


if __name__ == "__main__":
    main()
