"""Reference-region pretraining and shared metric helpers for the direct model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from .metrics import Fmax
from .model import DirectFunctionQuery


def atomic_json(path: Path, value: dict | list) -> None:
    """Publish completed records without exposing partially written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    """Bind results to immutable input and checkpoint files."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProteinRegions(Dataset):
    """Keep the original labelled-reference-region training groups intact."""

    def __init__(self, folder: Path) -> None:
        xs, ys, metadata = [], [], []
        for path in sorted(folder.glob("shard_*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            xs.append(shard["x"])
            ys.append(shard["y"])
            metadata.extend(shard["meta"])
        if not xs:
            raise FileNotFoundError(folder)
        self.x = torch.cat(xs).float()
        self.y = torch.cat(ys).float()
        grouped = defaultdict(list)
        for index, row in enumerate(metadata):
            grouped[row["accession"]].append(index)
        self.groups = list(grouped.values())
        if len(metadata) != len(self.x):
            raise ValueError("cache metadata size mismatch")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.groups[index]
        return self.x[rows], self.y[rows]


def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the same region padding as the original function-query trainer."""
    count = max(len(x) for x, _ in batch)
    x = torch.zeros(len(batch), count, 2560)
    y = torch.zeros(len(batch), count, 256)
    valid = torch.zeros(len(batch), count, dtype=torch.bool)
    for i, (features, labels) in enumerate(batch):
        x[i, :len(features)] = features
        y[i, :len(features)] = labels
        valid[i, :len(features)] = True
    return x, y, valid


def contrast_loss(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Vectorize the original per-pair, per-direction exclusive-label margin loss."""
    positive = targets > .5
    exclusive = positive[:, :, None, :] & ~positive[:, None, :, :]
    pairs = valid[:, :, None] & valid[:, None, :]
    pairs &= ~torch.eye(valid.shape[1], device=valid.device, dtype=torch.bool)[None]
    counts = exclusive.sum(-1)
    difference = logits[:, :, None, :] - logits[:, None, :, :]
    means = (F.softplus(2.0 - difference) * exclusive).sum(-1) / counts.clamp_min(1)
    selected = pairs & (counts > 0)
    return means[selected].mean() if selected.any() else logits.sum() * 0


def build_model(seed: int = 42) -> DirectFunctionQuery:
    """Initialize the paper's direct query without a baseline checkpoint."""
    torch.manual_seed(seed)
    return DirectFunctionQuery()


@torch.no_grad()
def validate_reference(model: nn.Module, dataset: Dataset, device: torch.device,
                       rank: int, world: int) -> dict:
    """Select checkpoints on complete reference-region validation without padded rows."""
    model.eval()
    thresholds = torch.linspace(0, 1, 101, device=device, dtype=torch.float64)
    stats = torch.zeros(3, 101, device=device, dtype=torch.float64)
    loader = DataLoader(Subset(dataset, list(range(rank, len(dataset), world))),
                        batch_size=64, collate_fn=collate, num_workers=0)
    for x, y, valid in loader:
        x, y, valid = x.to(device), y.to(device), valid.to(device)
        logits, _ = model(x, valid)
        probabilities, labels = logits[valid].sigmoid().double(), y[valid].bool()
        predicted = probabilities[:, None, :] >= thresholds[None, :, None]
        tp = (predicted & labels[:, None, :]).sum(-1)
        calls = predicted.sum(-1)
        positives = labels.sum(-1)[:, None]
        stats[0] += torch.where(calls > 0, tp / calls.clamp_min(1), 0).sum(0)
        stats[1] += torch.where(positives > 0, tp / positives.clamp_min(1), 0).sum(0)
        stats[2] += len(labels)
    dist.all_reduce(stats)
    precision, recall = stats[0] / stats[2], stats[1] / stats[2]
    score = 2 * precision * recall / (precision + recall).clamp_min(1e-20)
    index = int(score.argmax())
    return {"macro_fmax": float(score[index]), "threshold": index / 100,
            "precision": float(precision[index]), "recall": float(recall[index]),
            "regions": int(stats[2, 0])}


def train(args: argparse.Namespace, rank: int, world: int, device: torch.device) -> None:
    """Pretrain direct queries with the recorded optimizer, loss and budget."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = ProteinRegions(args.root / "output/region_mlp_baseline/cache/train")
    val_data = ProteinRegions(args.root / "output/region_mlp_baseline/cache/val")
    model = build_model(args.seed).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    positives = train_data.y.sum(0)
    weights = ((len(train_data.y) - positives) / positives.clamp_min(1)).clamp(1, 50).to(device)
    wrapped = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-2)
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, seed=args.seed)
    loader = DataLoader(train_data, batch_size=args.global_batch // world,
                        sampler=sampler, collate_fn=collate, num_workers=0, pin_memory=True)
    folder = args.output / args.mode
    folder.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        manifest = {"mode": args.mode, "epochs": args.epochs, "seed": args.seed,
                    "global_batch": args.global_batch, "world_size": world,
                    "train_proteins": len(train_data), "train_regions": len(train_data.x),
                    "val_proteins": len(val_data), "val_regions": len(val_data.x),
                    "lr": 3e-4, "weight_decay": .01, "rank_loss_weight": .25,
                    "output_head_init": "zero", "query_dim": 128,
                    "trainable_parameters": sum(p.numel() for p in trainable),
                    "base_checkpoint_sha256": None,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)), "numeric_threads": torch.get_num_threads(),
                    "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "scope": "supervised reference-region function-query pretraining"}
        atomic_json(folder / "manifest.json", manifest)
        print(json.dumps({"stage": "training", **manifest}), flush=True)
    print(json.dumps({"rank": rank, "gpu": torch.cuda.get_device_name(device),
                      "local_device": device.index, "training_batches": len(loader)}), flush=True)
    best, history = -1., []
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch - 1)
        wrapped.train()
        sums = torch.zeros(3, device=device)
        started = time.monotonic()
        for step, (x, y, valid) in enumerate(loader, 1):
            x, y, valid = x.to(device, non_blocking=True), y.to(device, non_blocking=True), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = wrapped(x, valid)
            bce = F.binary_cross_entropy_with_logits(logits[valid], y[valid], pos_weight=weights)
            ranking = contrast_loss(logits, y, valid)
            loss = bce + .25 * ranking
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            sums += torch.stack((bce.detach(), ranking.detach(), loss.new_tensor(1.)))
            if rank == 0 and step % 200 == 0:
                print(json.dumps({"mode": args.mode, "epoch": epoch, "step": step,
                                  "steps": len(loader), "loss": float(loss),
                                  "seconds": time.monotonic() - started}), flush=True)
            if args.smoke and step == 2:
                break
        dist.all_reduce(sums)
        metrics = validate_reference(model, val_data, device, rank, world)
        row = {"epoch": epoch, "train_bce": float(sums[0] / sums[2]),
               "train_ranking": float(sums[1] / sums[2]),
               "seconds": time.monotonic() - started, **metrics}
        if rank == 0:
            print(json.dumps({"mode": args.mode, **row}), flush=True)
            history.append(row)
            atomic_json(folder / "history.json", history)
            saved = {"state_dict": model.state_dict(), "mode": args.mode, "dim": 128,
                     "epoch": epoch, "seed": args.seed, "val": metrics,
                     "scope": "reference-region function-query initialization for continuation"}
            if metrics["macro_fmax"] > best + 1e-6:
                best = metrics["macro_fmax"]
                torch.save(saved, folder / "best.pt.tmp")
                (folder / "best.pt.tmp").replace(folder / "best.pt")
            torch.save(saved, folder / "last.pt.tmp")
            (folder / "last.pt.tmp").replace(folder / "last.pt")
        dist.barrier()
        if args.smoke:
            break


def combine_metric(state: Fmax, device: torch.device) -> dict:
    """Reduce exact streaming metric sufficient statistics across all GPUs."""
    value = np.vstack((state.p, state.r, state.tp, state.fp, state.fn,
                       np.full(101, state.n)))
    tensor = torch.as_tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    value = tensor.cpu().numpy()
    state.p, state.r, state.tp, state.fp, state.fn = value[:5]
    state.n = int(value[5, 0])
    result = state.result()
    p, r = state.p / max(state.n, 1), state.r / max(state.n, 1)
    result["macro_f1_by_threshold"] = np.divide(2 * p * r, p + r,
                                                out=np.zeros(101), where=p + r > 0).tolist()
    return result


def main() -> None:
    """Pretrain direct queries on reference regions using the recorded four-GPU recipe."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--global-batch', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.mode = 'direct'
    torch.set_num_threads(1)
    local = int(os.environ['LOCAL_RANK'])
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4 or args.global_batch % world:
        raise ValueError('The recorded recipe requires four allocated GPUs and a divisible global batch')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.backends.cuda.enable_cudnn_sdp(False)
    dist.init_process_group('nccl')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        train(args, rank, world, device)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
