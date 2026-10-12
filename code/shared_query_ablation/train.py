"""
Training script for the dual-query decoupled region--function model.

Two query streams (segmentation + function) over a shared residue memory,
fused into a unified protein representation (see model.py and README.md).
"""

import argparse
import os
import sys

# Some conda-forge Python builds expose a nonstandard version string that old
# ``platform.py`` releases cannot parse when PyTorch imports Triton.
sys.version = sys.version.replace(" | packaged by conda-forge | ", " ")

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from dataset import EndToEndDomainFunctionDataset, collate_fn
from lightning_module import DualQueryLightning


DATA_ROOT = "/geniusland/home/huyaxuan/code/region_function_old/data/dataset_v2/end_to_end_domain_function_v1"

# RTX 4090 Tensor Cores: substantially faster matmuls with stable training
# behavior for this architecture.
torch.set_float32_matmul_precision("high")


def train(
    train_jsonl=None,
    val_jsonl=None,
    esm_feature_root=None,
    data_root="/geniusland/home/huyaxuan/code/region_function_old",
    num_classes=256,
    batch_size=16,
    max_epochs=60,
    lr=1e-4,
    num_queries=10,
    num_dn_queries=4,
    num_gt_func_queries=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    rep_dim=256,
    task_loss_weight=30.0,
    rep_loss_weight=5.0,
    gt_func_loss_weight=10.0,
    dn_loss_weight=1.0,
    domain_sel_loss_weight=0.5,
    num_workers=8,
    gpus="0,1,2,3",
    checkpoint_dir=None,
    resume_from_checkpoint=None,
    max_len=1495,
    unmatched_mask_weight=0.1,
    objectness_loss_weight=1.0,
    mask_loss_weight=1.5,
    validity_loss_weight=1.0,
    mask_threshold=0.5,
    objectness_threshold=0.3,
    accumulate_grad_batches=2,
    precision="bf16-mixed",
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    # v2 fix: curriculum + split LR + warmup
    curriculum_coarse_end=12,
    curriculum_ramp_epochs=5,
    curriculum_freeze_encoder=True,
    lr_head=3e-4,
    lr_enc=1e-4,
    warmup_ratio=0.05,
    function_query_source="mask",
):
    pl.seed_everything(42, workers=True)
    if train_jsonl is None:
        train_jsonl = os.path.join(DATA_ROOT, "train.jsonl")
    if val_jsonl is None:
        val_jsonl = os.path.join(DATA_ROOT, "val.jsonl")
    if checkpoint_dir is None:
        checkpoint_dir = "weight/e2e_domain_cls_dual_query/"

    print(f"{'='*60}")
    print(f"Dual-query: Decoupled segmentation + function + unified representation")
    print(f"{'='*60}")
    print(f"  num_classes={num_classes}, rep_dim={rep_dim}")
    print(f"  queries={num_queries}, dn_queries={num_dn_queries}")
    print(f"  encoder={num_encoder_layers}, decoder={num_decoder_layers}")
    print(f"  batch_size={batch_size}, lr={lr}, max_len={max_len}")
    print(f"  task_loss_weight={task_loss_weight} (domain-level function)")
    print(f"  rep_loss_weight={rep_loss_weight} (unified representation)")
    print(f"  gt_func_loss_weight={gt_func_loss_weight} (GT-conditioned function)")
    print(f"  dn_loss_weight={dn_loss_weight}")
    print(f"  domain_sel_loss_weight={domain_sel_loss_weight}")
    print(f"  train: {train_jsonl}")
    print(f"  val:   {val_jsonl}")
    print(f"{'='*60}\n")

    # Datasets
    train_dataset = EndToEndDomainFunctionDataset(
        train_jsonl, num_classes=num_classes, max_len=max_len,
        esm_feature_root=esm_feature_root, data_root=data_root,
    )
    val_dataset = EndToEndDomainFunctionDataset(
        val_jsonl, num_classes=num_classes, max_len=max_len,
        esm_feature_root=esm_feature_root, data_root=data_root,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = DualQueryLightning(
        num_classes=num_classes,
        num_queries=num_queries,
        num_dn_queries=num_dn_queries,
        num_gt_func_queries=num_gt_func_queries,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        rep_dim=rep_dim,
        lr=lr,
        task_loss_weight=task_loss_weight,
        rep_loss_weight=rep_loss_weight,
        gt_func_loss_weight=gt_func_loss_weight,
        dn_loss_weight=dn_loss_weight,
        domain_sel_loss_weight=domain_sel_loss_weight,
        unmatched_mask_weight=unmatched_mask_weight,
        objectness_loss_weight=objectness_loss_weight,
        mask_loss_weight=mask_loss_weight,
        validity_loss_weight=validity_loss_weight,
        mask_threshold=mask_threshold,
        objectness_threshold=objectness_threshold,
        curriculum_coarse_end=curriculum_coarse_end,
        curriculum_ramp_epochs=curriculum_ramp_epochs,
        curriculum_freeze_encoder=curriculum_freeze_encoder,
        lr_head=lr_head,
        lr_enc=lr_enc,
        warmup_ratio=warmup_ratio,
        function_query_source=function_query_source,
    )

    # Callbacks
    cls_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_fmax-{epoch:02d}-{val_Fmax:.6f}',
        monitor='val_Fmax', mode='max', save_top_k=1, save_weights_only=True,
    )
    domain_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_domain-{epoch:02d}-{val_domain_IoU:.6f}',
        monitor='val_domain_IoU', mode='max', save_top_k=1, save_weights_only=True,
    )
    rep_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_rep-{epoch:02d}-{val_protein_Fmax:.6f}',
        monitor='val_protein_Fmax', mode='max', save_top_k=1, save_weights_only=True,
    )
    joint_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_joint-{epoch:02d}-{val_joint_score:.6f}',
        monitor='val_joint_score', mode='max', save_top_k=3, save_last=True,
    )
    e2e_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_e2e-{epoch:02d}-{val_e2e_domain_Fmax:.6f}',
        monitor='val_e2e_domain_Fmax', mode='max', save_top_k=1, save_weights_only=True,
    )
    tri_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_tri-{epoch:02d}-{val_tri_score:.6f}',
        monitor='val_tri_score', mode='max', save_top_k=3, save_last=False,
    )
    early_stop_callback = EarlyStopping(
        monitor='val_tri_score', patience=30, mode='max', verbose=True,
    )

    # Trainer
    trainer_kwargs = {
        'max_epochs': max_epochs,
        'default_root_dir': os.path.dirname(os.path.abspath(checkpoint_dir)),
        'callbacks': [cls_checkpoint, domain_checkpoint, rep_checkpoint,
                      joint_checkpoint, e2e_checkpoint, tri_checkpoint,
                      early_stop_callback],
        'log_every_n_steps': 10,
        'enable_progress_bar': True,
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': accumulate_grad_batches,
        'precision': precision,
        'limit_train_batches': limit_train_batches,
        'limit_val_batches': limit_val_batches,
    }
    if resume_from_checkpoint:
        from recovery_diagnostics import RecoveryDiagnostics
        trainer_kwargs['callbacks'].append(RecoveryDiagnostics(
            resume_from_checkpoint, os.path.dirname(os.path.abspath(checkpoint_dir)),
        ))

    if isinstance(gpus, str):
        gpu_list = [int(x.strip()) for x in gpus.split(',')]
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = gpu_list
        if len(gpu_list) > 1:
            trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=True)
    elif isinstance(gpus, int) and gpus > 0:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = gpus
    else:
        trainer_kwargs['accelerator'] = 'cpu'

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
    return model, trainer


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_queries", type=int, default=10)
    p.add_argument("--num_dn_queries", type=int, default=4)
    p.add_argument("--num_gt_func_queries", type=int, default=4)
    p.add_argument("--num_encoder_layers", type=int, default=3)
    p.add_argument("--num_decoder_layers", type=int, default=3)
    p.add_argument("--rep_dim", type=int, default=256)
    p.add_argument("--task_loss_weight", type=float, default=30.0)
    p.add_argument("--rep_loss_weight", type=float, default=5.0)
    p.add_argument("--gt_func_loss_weight", type=float, default=10.0)
    p.add_argument("--dn_loss_weight", type=float, default=1.0)
    p.add_argument("--domain_sel_loss_weight", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1495)
    p.add_argument("--unmatched_mask_weight", type=float, default=0.1)
    p.add_argument("--objectness_loss_weight", type=float, default=1.0)
    p.add_argument("--mask_loss_weight", type=float, default=1.5)
    p.add_argument("--validity_loss_weight", type=float, default=1.0)
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--objectness_threshold", type=float, default=0.3)
    p.add_argument("--accumulate_grad_batches", type=int, default=2)
    p.add_argument("--precision", type=str, default="bf16-mixed")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--esm_feature_root", type=str, default=None)
    p.add_argument("--data_root", type=str,
                   default="/geniusland/home/huyaxuan/code/region_function_old")
    p.add_argument("--train_jsonl", type=str, default=None)
    p.add_argument("--val_jsonl", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument('--resume_from_checkpoint', type=str, default=None)
    # v2 fix: curriculum + split LR + warmup
    p.add_argument("--curriculum_coarse_end", type=int, default=12)
    p.add_argument("--curriculum_ramp_epochs", type=int, default=5)
    p.add_argument("--no_freeze_encoder", dest="curriculum_freeze_encoder",
                   action="store_false", default=True,
                   help="disable freezing the shared encoder during the classification phase")
    p.add_argument("--lr_head", type=float, default=3e-4)
    p.add_argument("--lr_enc", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument(
        "--function_query_source", choices=("mask", "global"), default="mask",
        help="regional content used to initialize regular function queries",
    )
    args = p.parse_args()

    train(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        num_queries=args.num_queries,
        num_dn_queries=args.num_dn_queries,
        num_gt_func_queries=args.num_gt_func_queries,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        rep_dim=args.rep_dim,
        task_loss_weight=args.task_loss_weight,
        rep_loss_weight=args.rep_loss_weight,
        gt_func_loss_weight=args.gt_func_loss_weight,
        dn_loss_weight=args.dn_loss_weight,
        domain_sel_loss_weight=args.domain_sel_loss_weight,
        num_workers=args.num_workers,
        gpus=args.gpus,
        max_len=args.max_len,
        esm_feature_root=args.esm_feature_root,
        data_root=args.data_root,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        unmatched_mask_weight=args.unmatched_mask_weight,
        objectness_loss_weight=args.objectness_loss_weight,
        mask_loss_weight=args.mask_loss_weight,
        validity_loss_weight=args.validity_loss_weight,
        mask_threshold=args.mask_threshold,
        objectness_threshold=args.objectness_threshold,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        curriculum_coarse_end=args.curriculum_coarse_end,
        curriculum_ramp_epochs=args.curriculum_ramp_epochs,
        curriculum_freeze_encoder=args.curriculum_freeze_encoder,
        lr_head=args.lr_head,
        lr_enc=args.lr_enc,
        warmup_ratio=args.warmup_ratio,
        function_query_source=args.function_query_source,
    )
