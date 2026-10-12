#!/usr/bin/env python3
"""Cache one frozen forward pass per protein. Never starts training."""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
from common import read_json, resolve, sha256, source_hashes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default="/geniusland/home/huyaxuan/code/region_function_multimodal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="Smoke subset; use a separate --output directory")
    parser.add_argument("--output", help="Override cache path")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if cfg["split"] not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if cfg["precision"] not in ("float32", "bfloat16"):
        raise ValueError("Unsupported precision")
    output = resolve(args.project_root, args.output or cfg["cache"])
    code = resolve(args.project_root, cfg["model_code"]).resolve()
    dataset_path = resolve(args.project_root, cfg["dataset"]).resolve()
    checkpoint = resolve(args.project_root, cfg["checkpoint"]).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Cache exists; use --resume or a new directory: {output}")
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    sys.path.insert(0, str(code))
    dataset_module = importlib.import_module("dataset")
    lightning_module = importlib.import_module("lightning_module")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    ds = dataset_module.EndToEndDomainFunctionDataset(
        str(dataset_path), num_classes=cfg["num_classes"], max_len=cfg["max_len"],
        data_root=str(resolve(args.project_root, cfg["data_root"])),
        esm_feature_root=cfg.get("esm_feature_root"),
    )
    count = min(args.limit, len(ds)) if args.limit is not None else len(ds)
    if count == 0:
        raise ValueError("Empty dataset")
    accessions = [str(record["uniprot_acc"]) for record in ds.samples[:count]]
    if len(set(accessions)) != count:
        raise ValueError("Duplicate accessions in cohort")
    # Hashing a checkpoint is CPU/I/O work: invoke this program through run_limited.sh.
    signature = {
        "format": 1, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "dataset": str(dataset_path), "dataset_sha256": sha256(dataset_path),
        "model_source": source_hashes(code), "cache_code_sha256": sha256(__file__),
        "max_len": cfg["max_len"], "num_classes": cfg["num_classes"], "precision": cfg["precision"],
        "data_root": str(resolve(args.project_root, cfg["data_root"])),
        "esm_feature_root": cfg.get("esm_feature_root"), "split": cfg["split"],
        "seed": cfg["seed"], "device": args.device, "torch": torch.__version__,
        "numpy": np.__version__, "limit": args.limit, "count": count,
        "cohort_sha256": __import__("hashlib").sha256("\n".join(accessions).encode()).hexdigest(),
    }
    protocol_path = output / "protocol.json"
    if output.exists() and not protocol_path.exists():
        raise ValueError("Existing directory has no cache protocol; use a new directory")
    if protocol_path.exists():
        prior = read_json(protocol_path)
        if prior["signature"] != signature:
            raise ValueError("Cache provenance differs. Use a new cache directory.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    write_json(protocol_path, {"signature": signature, "complete": False})
    model = None
    index = []
    for i, accession in enumerate(accessions):
        record = ds.samples[i]
        feature_path = (Path(cfg["esm_feature_root"]) / Path(record["esm_path"]).name
                        if cfg.get("esm_feature_root") else resolve(signature["data_root"], record["esm_path"]))
        feature_hash = sha256(feature_path)
        target = output / "samples" / f"{i:06d}.npz"
        if target.exists() and args.resume:
            with np.load(target, allow_pickle=False) as saved:
                if str(saved["accession"].item()) != accession or str(saved["feature_sha256"].item()) != feature_hash:
                    raise ValueError(f"Cached feature/accession changed: {accession}")
        else:
            if model is None:
                loaded = lightning_module.DualQueryLightning.load_from_checkpoint(
                    str(checkpoint), map_location="cpu", strict=True)
                model = loaded.model.eval().to(args.device)
            sample = ds[i]
            length = int(sample["seq_len"])
            if length != min(len(record["sequence"]), cfg["max_len"]):
                raise ValueError(f"Sequence / ESM length mismatch: {accession}")
            # Keep the original padded model input to reproduce the dataset convention.
            features = sample["residue_feat"].unsqueeze(0).to(args.device)
            attention = sample["attn_mask"].unsqueeze(0).to(args.device)
            with torch.inference_mode(), torch.autocast(
                    device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                    enabled=cfg["precision"] == "bfloat16"):
                out = model(features, attention, training=False)
            def array(name):
                value = out[name][0].detach().float().cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite {name}: {accession}")
                return value
            infos = sample["domain_infos"]
            values = dict(
                accession=np.asarray(accession), length=np.asarray(length),
                original_length=np.asarray(len(record["sequence"])),
                feature_sha256=np.asarray(feature_hash),
                mask_logits=array("mask_logits")[:, :length], func_logits=array("func_logits"),
                objectness_logits=array("objectness_logits"), protein_logits=array("protein_logits"),
                gt_masks=np.stack([d["mask"][:length].numpy().astype(bool) for d in infos]) if infos else np.zeros((0, length), bool),
                gt_labels=np.stack([d["label"].numpy() for d in infos]) if infos else np.zeros((0, cfg["num_classes"]), np.float32),
                gt_valid=np.asarray([bool(d["valid"]) for d in infos]),
                gt_source_valid=np.asarray([bool(d.get("source_function_valid", d["valid"])) for d in infos]),
                protein_label=sample["protein_label"].numpy(),
                dropped_invisible=np.asarray(sample.get("dropped_invisible_domains", 0)),
            )
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **values)
            os.replace(temporary, target)
        index.append({"accession": accession, "file": str(target.relative_to(output)), "sha256": sha256(target)})
        if (i + 1) % 100 == 0 or i + 1 == count:
            print(f"Cached {i + 1}/{count}", flush=True)
    write_json(output / "index.json", index)
    write_json(protocol_path, {"signature": signature, "complete": True, "index_sha256": sha256(output / "index.json")})
    print(f"Complete cache: {output}")


if __name__ == "__main__":
    main()
