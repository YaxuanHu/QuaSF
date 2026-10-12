"""
Training script for end-to-end domain segmentation + protein-level classification.
"""

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from dataset import EndToEndDomainFunctionDataset, collate_fn
from lightning_module import E2EProteinClsLightning


DATA_ROOT = "/geniusland/home/huyaxuan/code/region_function_old/data/dataset_v2/end_to_end_domain_function_v1"


def train(
    train_jsonl=None,
    val_jsonl=None,
    esm_feature_root=None,
    num_classes=256,
    batch_size=16,
    max_epochs=60,
    lr=1e-4,
    num_queries=8,
    num_dn_queries=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    task_loss_weight=50.0,
    dn_loss_weight=1.0,
    domain_sel_loss_weight=0.5,
    num_workers=4,
    gpus="0,1,2,3",
    checkpoint_dir=None,
    max_len=1024,
):
    if train_jsonl is None:
        train_jsonl = os.path.join(DATA_ROOT, "train.jsonl")
    if val_jsonl is None:
        val_jsonl = os.path.join(DATA_ROOT, "val.jsonl")
    if checkpoint_dir is None:
        checkpoint_dir = f"weight/e2e_protein_cls/tlw_{int(task_loss_weight)}/"

    print(f"{'='*60}")
    print(f"End-to-End: Segmentation + Protein-level Classification")
    print(f"{'='*60}")
    print(f"  num_classes={num_classes}")
    print(f"  queries={num_queries}, dn_queries={num_dn_queries}")
    print(f"  encoder={num_encoder_layers}, decoder={num_decoder_layers}")
    print(f"  batch_size={batch_size}, lr={lr}, max_len={max_len}")
    print(f"  task_loss_weight={task_loss_weight}")
    print(f"  dn_loss_weight={dn_loss_weight}")
    print(f"  domain_sel_loss_weight={domain_sel_loss_weight}")
    print(f"  train: {train_jsonl}")
    print(f"  val:   {val_jsonl}")
    print(f"{'='*60}\n")

    # Datasets
    train_dataset = EndToEndDomainFunctionDataset(
        train_jsonl, num_classes=num_classes, max_len=max_len,
        esm_feature_root=esm_feature_root,
    )
    val_dataset = EndToEndDomainFunctionDataset(
        val_jsonl, num_classes=num_classes, max_len=max_len,
        esm_feature_root=esm_feature_root,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = E2EProteinClsLightning(
        num_classes=num_classes,
        num_queries=num_queries,
        num_dn_queries=num_dn_queries,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        lr=lr,
        task_loss_weight=task_loss_weight,
        dn_loss_weight=dn_loss_weight,
        domain_sel_loss_weight=domain_sel_loss_weight,
    )

    # Callbacks
    cls_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_fmax-{epoch:02d}-{val_Fmax:.6f}',
        monitor='val_Fmax', mode='max', save_top_k=3, save_last=True,
    )
    domain_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_domain-{epoch:02d}-{val_domain_IoU:.6f}',
        monitor='val_domain_IoU', mode='max', save_top_k=3,
    )
    early_stop_callback = EarlyStopping(
        monitor='val_Fmax', patience=30, mode='max', verbose=True,
    )

    # Trainer
    trainer_kwargs = {
        'max_epochs': max_epochs,
        'callbacks': [cls_checkpoint, domain_checkpoint, early_stop_callback],
        'log_every_n_steps': 10,
        'enable_progress_bar': True,
        'gradient_clip_val': 1.0,
    }

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
    trainer.fit(model, train_loader, val_loader)
    return model, trainer


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_queries", type=int, default=8)
    p.add_argument("--num_dn_queries", type=int, default=4)
    p.add_argument("--num_encoder_layers", type=int, default=3)
    p.add_argument("--num_decoder_layers", type=int, default=3)
    p.add_argument("--task_loss_weight", type=float, default=50.0)
    p.add_argument("--dn_loss_weight", type=float, default=1.0)
    p.add_argument("--domain_sel_loss_weight", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--esm_feature_root", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    args = p.parse_args()

    train(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        num_queries=args.num_queries,
        num_dn_queries=args.num_dn_queries,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        task_loss_weight=args.task_loss_weight,
        dn_loss_weight=args.dn_loss_weight,
        domain_sel_loss_weight=args.domain_sel_loss_weight,
        num_workers=args.num_workers,
        gpus=args.gpus,
        max_len=args.max_len,
        esm_feature_root=args.esm_feature_root,
        checkpoint_dir=args.checkpoint_dir,
    )
