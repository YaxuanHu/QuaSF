"""Training script for EguchiCNN on HeMeNet dataset (PyTorch Lightning).

Usage:
    # 单卡
    python -m baselines.EguchiCNN.trainer --gpus 0

    # 多卡
    python -m baselines.EguchiCNN.trainer --gpus 0,1,2,3
"""
import os
import sys
import json
import argparse
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

import torch

# Enable TF32 for faster matmul/conv on Ampere/Ada GPUs
torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))

from lightning_module import EguchiCNNLightningModule
from dataset import EguchiCNNDataset, EguchiCNNH5Dataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
LOG = logging.getLogger(__name__)


def parse_gpus(gpu_str: str) -> list[int]:
    """Parse GPU string like '0,1,2,3' or '0-3' into list of ints."""
    gpu_str = gpu_str.strip()
    if not gpu_str:
        return [0]
    ids = []
    for part in gpu_str.split(','):
        part = part.strip()
        if '-' in part:
            tokens = part.split('-')
            start, end = int(tokens[0]), int(tokens[1])
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))
    return ids


def main():
    parser = argparse.ArgumentParser(description='Train EguchiCNN on HeMeNet data')
    parser.add_argument('--train_csv', type=str,
                        default='data/HeMeNet/train_dataset_with_structure.csv')
    parser.add_argument('--val_csv', type=str,
                        default='baselines/results/merizo_evaluated_proteins.csv')
    parser.add_argument('--structures_dir', type=str,
                        default='data/HeMeNet/structures')
    parser.add_argument('--output_dir', type=str,
                        default='baselines/EguchiCNN/checkpoints')
    parser.add_argument('--cache_dir', type=str, default='',
                        help='Directory with precomputed .pt feature files')
    parser.add_argument('--hdf5_dir', type=str, default='',
                        help='Directory containing train.h5/val.h5/test.h5 precomputed features')
    parser.add_argument('--epochs', type=int, default=160,
                        help='Total epochs (paper: 160)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (paper: 0.001)')
    parser.add_argument('--lr_step_epoch', type=int, default=60,
                        help='Epoch to reduce LR (paper: 60)')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='LR reduction factor (paper: 0.001 -> 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability (paper: 0.1)')
    parser.add_argument('--max_len', type=int, default=1024,
                        help='Max sequence length / contact map size (paper: 512)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes (linker + max domains)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (paper: 64)')
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', default='4,5',
    # parser.add_argument('--gpus', default='4,5,6,7',
                        help='GPUs to use (e.g. "0,1" or "0-3")')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = {
        'max_len': args.max_len,
        'num_classes': args.num_classes,
        'dropout': args.dropout,
        'lr': args.lr,
        'lr_step_epoch': args.lr_step_epoch,
        'lr_factor': args.lr_factor,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Datasets
    if args.hdf5_dir:
        LOG.info("Loading training dataset from HDF5...")
        train_dataset = EguchiCNNH5Dataset(
            csv_path=args.train_csv,
            hdf5_path=os.path.join(args.hdf5_dir, 'train.h5'),
            max_len=args.max_len,
        )
        LOG.info("Loading validation dataset from HDF5...")
        val_dataset = EguchiCNNH5Dataset(
            csv_path=args.val_csv,
            hdf5_path=os.path.join(args.hdf5_dir, 'val.h5'),
            max_len=args.max_len,
        )
    else:
        LOG.info("Loading training dataset...")
        train_dataset = EguchiCNNDataset(
            csv_path=args.train_csv,
            structures_dir=args.structures_dir,
            max_len=args.max_len,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        LOG.info("Loading validation dataset...")
        val_dataset = EguchiCNNDataset(
            csv_path=args.val_csv,
            structures_dir=args.structures_dir,
            max_len=args.max_len,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Lightning module
    model = EguchiCNNLightningModule(
        max_len=args.max_len,
        num_classes=args.num_classes,
        dropout=args.dropout,
        lr=args.lr,
        lr_step_epoch=args.lr_step_epoch,
        lr_factor=args.lr_factor,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info(f"Model parameters: {n_params:,}")

    # GPU setup
    devices = parse_gpus(args.gpus)
    strategy = "ddp" if len(devices) > 1 else "auto"
    LOG.info(f"Using GPUs: {devices}, strategy: {strategy}")

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=devices,
        strategy=strategy,
        accumulate_grad_batches=args.accumulation_steps,
        callbacks=[ckpt_callback, lr_monitor],
        default_root_dir=args.output_dir,
        log_every_n_steps=50,
        precision='16-mixed',
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    LOG.info(f"Training complete. Best checkpoint: {ckpt_callback.best_model_path}")
    LOG.info(f"Best val_loss: {ckpt_callback.best_model_score}")


if __name__ == '__main__':
    main()
