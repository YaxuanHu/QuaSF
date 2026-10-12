"""Evaluate all joint checkpoints on validation, then only selected ones on test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .paper_train import restore
from .common import runs
from .data import EndToEndDomainFunctionDataset, collate_fn
from .decoders import Prediction
from .evaluate import fixed_macro_f1
from .metrics import Fmax, PROTOCOL, add_rf, segmentation
from .pretrain import atomic_json, combine_metric, sha256


def score_checkpoint(args: argparse.Namespace, checkpoint: Path, output: Path,
                     selection: dict | None, rank: int, device: torch.device) -> dict:
    """Shard proteins without padding and reduce sufficient statistics exactly."""
    manifest = json.loads((args.joint_output / args.mode / "manifest.json").read_text())
    mask_path, query_path = Path(manifest["mask_checkpoint"]), Path(manifest["query_checkpoint"])
    module = restore(args.root, query_path, mask_path, args.mode)
    module.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    model = module.model.to(device).eval()
    split_path = args.root / "data/dataset_v2/end_to_end_domain_function_v1" / f"{args.split}.jsonl"
    dataset = EndToEndDomainFunctionDataset(split_path, max_len=1495, data_root=str(args.root))
    size = min(len(dataset), args.limit) if args.limit else len(dataset)
    indices = list(range(rank, size, dist.get_world_size()))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)
    strict, matched, protein = Fmax(), Fmax(), Fmax()
    # n, GT union residues, weighted domain IoU, boundary F1 sum,
    # detection TP/FP/FN at .5 and .75, regional correct/comparisons/eligible/all-correct.
    totals = np.zeros(14, dtype=np.float64)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / f"predictions_rank{rank}.jsonl.tmp"
    completed = 0
    with temporary.open("w", encoding="utf-8") as sink, torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch["residue_feat"].to(device, non_blocking=True),
                               batch["attn_mask"].to(device, non_blocking=True), training=False)
            # Copy whole outputs once instead of synchronizing per region and class.
            result = {key: result[key].float().cpu() if result[key].is_floating_point()
                      else result[key].cpu() for key in (
                          "region_logits", "region_valid", "region_masks", "region_source_query_ids",
                          "mask_logits", "objectness_logits", "protein_logits")}
            for i, domains in enumerate(batch["domain_infos"]):
                length = batch["seq_len"][i]
                regional = []
                for position in range(result["region_logits"].shape[1]):
                    if not bool(result["region_valid"][i, position]):
                        continue
                    region = result["region_masks"][i, position, :length].numpy().astype(bool)
                    query = int(result["region_source_query_ids"][i, position])
                    values = result["region_logits"][i, position].sigmoid().numpy()
                    confidence = result["mask_logits"][i, query, :length].sigmoid().numpy()
                    obj = float(result["objectness_logits"][i, query].sigmoid())
                    regional.append(Prediction(query, region, values, float(confidence[region].mean()), obj))
                gt_masks = np.stack([d["mask"][:length].numpy().astype(bool) for d in domains]) if domains else np.zeros((0, length), bool)
                labels = np.stack([d["label"].numpy() for d in domains]) if domains else np.zeros((0, 256), np.float32)
                valid = np.asarray([bool(d["valid"]) for d in domains])
                row, pairs = segmentation(regional, gt_masks, length, (.5, .75), 20)
                row["accession"] = batch["uniprot_acc"][i]
                row["matches"] = {str(level): pairs[level] for level in (.5, .75)}
                row["strict_counts"] = add_rf(strict, matched, regional, labels, valid, pairs[.5])
                protein.add(result["protein_logits"][i].sigmoid().numpy(), batch["protein_label"][i].numpy())
                totals[:4] += [1, row["gt_residues"], row["domain_iou"] * row["gt_residues"], row["boundary_f1"]]
                totals[4:10] += [row[f"domain_{name}@{level}"] for level in (.5, .75) for name in ("tp", "fp", "fn")]
                observed = [(p, g) for p, g, _ in pairs[.5] if valid[g]]
                margins = []
                for pos, (left, left_gt) in enumerate(observed):
                    for right, right_gt in observed[pos + 1:]:
                        left_only = (labels[left_gt] > .5) & (labels[right_gt] < .5)
                        right_only = (labels[right_gt] > .5) & (labels[left_gt] < .5)
                        delta = regional[left].function_probs - regional[right].function_probs
                        margins.extend(delta[left_only].tolist())
                        margins.extend((-delta[right_only]).tolist())
                if margins:
                    totals[10:] += [sum(v > 0 for v in margins), len(margins), 1, int(all(v > 0 for v in margins))]
                row["predictions"] = [dict(domain_id=j + 1, source_query_id=p.query_id,
                                           segments=runs(p.mask), function_probs=p.function_probs.tolist(),
                                           mask_score=p.mask_score, objectness=p.objectness)
                                      for j, p in enumerate(regional)]
                sink.write(json.dumps(row, allow_nan=False) + "\n")
                completed += 1
            if completed % 400 < args.batch_size:
                print(json.dumps({"mode": args.mode, "split": args.split, "rank": rank,
                                  "checkpoint": checkpoint.name, "proteins": completed}), flush=True)
    temporary.replace(output / f"predictions_rank{rank}.jsonl")
    values = torch.as_tensor(totals, device=device)
    dist.all_reduce(values)
    totals = values.cpu().numpy()
    if int(totals[0]) != size:
        raise ValueError(f"Incomplete distributed evaluation: {int(totals[0])} of {size} proteins")
    summary = {"mode": args.mode, "split": args.split, "proteins": int(totals[0]),
               "partial": bool(args.limit), "protocol": PROTOCOL,
               "joint_checkpoint": str(checkpoint), "joint_checkpoint_sha256": sha256(checkpoint),
               "mask_checkpoint_sha256": sha256(mask_path), "query_checkpoint_sha256": sha256(query_path),
               "split_sha256": sha256(split_path), "evaluation_batch_per_gpu": args.batch_size,
               "evaluator_sha256": sha256(Path(__file__)),
               "joint_model_sha256": sha256(Path(__file__).with_name("model.py")),
               "identity_iou": float(totals[2] / totals[1]) if totals[1] else None,
               "boundary_macro_f1": float(totals[3] / totals[0]),
               "strict_rf": combine_metric(strict, device),
               "matched_function": combine_metric(matched, device),
               "protein_function": combine_metric(protein, device),
               "regional_preference": {"correct": int(totals[10]), "comparisons": int(totals[11]),
                                       "eligible_proteins": int(totals[12]), "proteins_all_correct": int(totals[13]),
                                       "fraction_correct": float(totals[10] / totals[11]) if totals[11] else None}}
    summary["detection"] = {}
    for offset, level in [(4, ".5"), (7, ".75")]:
        tp, fp, fn = totals[offset:offset + 3]
        summary["detection"][level] = {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                                        "micro_f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.}
    if selection is None:
        summary["selected_threshold"] = summary["strict_rf"]["macro_threshold"]
    else:
        if selection["split"] != "val" or selection["partial"] or selection["joint_checkpoint_sha256"] != summary["joint_checkpoint_sha256"]:
            raise ValueError("Test selection must be bound to complete validation of this checkpoint")
        summary["selected_threshold"] = selection["selected_threshold"]
        summary["strict_rf_f1_at_val_threshold"] = fixed_macro_f1(strict, selection["selected_threshold"])
    if rank == 0:
        atomic_json(output / "summary.json", summary)
        print(json.dumps({"output": str(output), "strict_rf": summary["strict_rf"]["macro_fmax"]}), flush=True)
    dist.barrier()
    del module, model
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    """Require validation selection before held-out test access."""
    parser = argparse.ArgumentParser()
    for name in ("root", "joint-output", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=["direct"], default="direct")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="partial smoke only, cannot select a model")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.enable_cudnn_sdp(False)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    if dist.get_world_size() != 4:
        raise ValueError("Four allocated GPUs are required")
    rank, device = dist.get_rank(), torch.device("cuda", local)
    folder = args.output / args.mode
    if args.limit:
        folder = folder / f"partial_{args.limit}"
    if args.split == "val":
        checkpoints = sorted((args.joint_output / args.mode / "checkpoints").glob("epoch=*.ckpt"))
        if len(checkpoints) != 6 and not args.limit:
            raise ValueError(f"Expected six completed epoch checkpoints, found {len(checkpoints)}")
        if not checkpoints:
            raise ValueError("No epoch checkpoint is available")
        candidates = []
        for checkpoint in checkpoints:
            output = folder / "val" / checkpoint.stem
            existing = output / "summary.json"
            if existing.exists():
                summary = json.loads(existing.read_text())
                if (summary["joint_checkpoint_sha256"] != sha256(checkpoint)
                        or summary["evaluation_batch_per_gpu"] != args.batch_size
                        or summary["evaluator_sha256"] != sha256(Path(__file__))):
                    raise ValueError("Existing evaluation checkpoint hash mismatch")
            else:
                summary = score_checkpoint(args, checkpoint, output, None, rank, device)
            candidates.append(summary)
        if rank == 0 and not args.limit:
            selected = max(candidates, key=lambda row: row["strict_rf"]["macro_fmax"])
            atomic_json(folder / "selected_val.json", selected)
            atomic_json(folder / "validation_candidates.json", candidates)
    else:
        selection = json.loads((folder / "selected_val.json").read_text())
        existing = folder / "test" / "summary.json"
        if existing.exists():
            summary = json.loads(existing.read_text())
            if (summary["joint_checkpoint_sha256"] != selection["joint_checkpoint_sha256"]
                    or summary["selected_threshold"] != selection["selected_threshold"]
                    or summary["evaluation_batch_per_gpu"] != args.batch_size
                    or summary["evaluator_sha256"] != sha256(Path(__file__))):
                raise ValueError("Existing test evaluation belongs to another frozen selection or evaluator")
        else:
            score_checkpoint(args, Path(selection["joint_checkpoint"]), folder / "test", selection, rank, device)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
