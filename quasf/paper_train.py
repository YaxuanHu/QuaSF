"""Continue pretrained direct function queries with the recorded joint curriculum."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from .data import EndToEndDomainFunctionDataset, collate_fn
from .lightning_module import DualQueryLightning
from .train import JointCurriculum
from .pretrain import atomic_json, build_model, sha256


class Throughput(Callback):
    """Report completed batches so utilization tuning follows actual useful work."""

    def on_train_epoch_start(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        self.started = time.monotonic()
        self.samples = 0

    def on_train_batch_end(self, trainer: pl.Trainer, module: pl.LightningModule,
                           outputs: object, batch: dict, batch_idx: int) -> None:
        self.samples += len(batch["residue_feat"]) * trainer.world_size
        if trainer.is_global_zero and (batch_idx + 1) % 20 == 0:
            seconds = time.monotonic() - self.started
            print(json.dumps({"epoch": trainer.current_epoch, "completed_batches": batch_idx + 1,
                              "seconds": seconds, "proteins_per_second": self.samples / seconds,
                              "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)


def restore(root: Path, query_path: Path, mask_path: Path, mode: str) -> JointCurriculum:
    """Restore the pretrained direct query and the recorded segmentation source."""
    if mode != "direct":
        raise ValueError("The paper release supports the direct model only")
    source = torch.load(mask_path, map_location="cpu", weights_only=False)
    allowed = set(DualQueryLightning.__init__.__code__.co_varnames) - {"self"}
    hparams = {k: v for k, v in source["hyper_parameters"].items() if k in allowed}
    module = JointCurriculum(hparams, mask_threshold=.3, query_warmup_epochs=1)
    module.load_state_dict(source["state_dict"], strict=True)
    query = build_model()
    saved = torch.load(query_path, map_location="cpu", weights_only=False)
    query.load_state_dict(saved["state_dict"], strict=True)
    module.attach_query(query)
    return module


def main() -> None:
    """Continue the direct query using the recorded six-epoch recipe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["direct"], default="direct")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--query-checkpoint", type=Path,
                        help="Pretrained direct query; defaults to the recorded reference-region checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    pl.seed_everything(42, workers=True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    mask_path = args.root / "output/mask_condition_ablation/mask_resume_20260922_101745/checkpoints/best_tri-epoch=12-val_tri_score=0.779311.ckpt"
    query_path = args.query_checkpoint or (args.root / "output/query_no_mlp_ablation_20260928" / args.mode / "best.pt")
    module = restore(args.root, query_path, mask_path, args.mode)
    data = args.root / "data/dataset_v2/end_to_end_domain_function_v1"
    train = EndToEndDomainFunctionDataset(data / "train.jsonl", max_len=1495, data_root=str(args.root))
    val = EndToEndDomainFunctionDataset(data / "val.jsonl", max_len=1495, data_root=str(args.root))
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=4, collate_fn=collate_fn, pin_memory=True)
               for dataset, shuffle in [(train, True), (val, False)]]
    accumulation = 128 // (4 * args.batch_size)
    if accumulation < 1 or accumulation * 4 * args.batch_size != 128:
        raise ValueError("batch must preserve effective global batch 128")
    output = args.output / args.mode
    # Each DDP worker may start this entry point; only local rank zero writes metadata.
    import os
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        atomic_json(output / "manifest.json", {
            "mode": args.mode, "mask_checkpoint": str(mask_path),
            "mask_sha256": sha256(mask_path), "query_checkpoint": str(query_path),
            "query_sha256": sha256(query_path), "seed": 42, "epochs": 6,
            "warmup": 1, "devices": 4, "batch_per_gpu": args.batch_size,
            "global_batch": 128, "accumulation": accumulation, "precision": "bf16-mixed",
            "train": str(data / "train.jsonl"), "val": str(data / "val.jsonl"),
            "smoke": args.smoke,
            "query_aggregation": "batched scatter_reduce; original outputs preserved",
            "selection": "All six checkpoints retained for unified validation-only selection"
        })
    checkpoints = ModelCheckpoint(dirpath=output / "checkpoints", save_last=True,
                                  save_top_k=-1, every_n_epochs=1)
    trainer = pl.Trainer(max_epochs=6, accelerator="gpu", devices=4,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         precision="bf16-mixed", callbacks=[checkpoints, Throughput()],
                         default_root_dir=output, gradient_clip_val=1,
                         accumulate_grad_batches=accumulation, fast_dev_run=args.smoke,
                         log_every_n_steps=50)
    trainer.fit(module, *loaders)


if __name__ == "__main__":
    main()
