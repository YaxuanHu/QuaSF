"""Fast batched evaluation of EguchiCNN using precomputed HDF5 features."""
import os
import sys
import json
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import EguchiCNN
from dataset import EguchiCNNH5Dataset, collate_fn
from evaluate_metrics import evaluate_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
LOG = logging.getLogger(__name__)


def load_model(checkpoint_dir, device):
    """Load trained EguchiCNN model from checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    model = EguchiCNN(
        max_len=config['max_len'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.1),
    )

    ckpt_path = os.path.join(checkpoint_dir, 'best.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, 'last.ckpt')

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        LOG.info(f"Loaded model from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    model.to(device)
    model.eval()
    return model, config


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info(f"EguchiCNN HDF5 Evaluation on {device}")

    model, config = load_model(args.checkpoint_dir, device)
    max_len = config.get('max_len', 512)

    dataset = EguchiCNNH5Dataset(
        csv_path=args.test_csv,
        hdf5_path=args.hdf5_path,
        max_len=max_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch['input'].to(device)      # [B, 1, L, L]
            labels = batch['labels']            # [B, L]
            mask = batch['mask']                # [B, L]
            n_res_list = batch['n_res']
            pdb_ids = batch['pdb_id']

            logits = model(x)                   # [B, C, L]
            preds = logits.argmax(dim=1).cpu().numpy()  # [B, L]
            labels_np = labels.cpu().numpy()

            for i in range(len(pdb_ids)):
                n_res = int(n_res_list[i])
                pdb_id = pdb_ids[i]

                if n_res <= 0:
                    continue

                if n_res <= max_len:
                    pred_labels = preds[i, :n_res]
                    gt_labels = labels_np[i, :n_res]
                else:
                    start = (n_res - max_len) // 2
                    pred_full = np.zeros(n_res, dtype=np.int64)
                    pred_full[start:start + max_len] = preds[i]
                    pred_labels = pred_full
                    gt_full = np.zeros(n_res, dtype=np.int64)
                    gt_full[start:start + max_len] = labels_np[i]
                    gt_labels = gt_full

                if gt_labels.max() == 0:
                    continue

                predictions.append((pdb_id, pred_labels))
                ground_truths.append((pdb_id, gt_labels))

            if (batch_idx + 1) % 50 == 0:
                LOG.info(f"  Processed {(batch_idx + 1) * args.batch_size} samples...")

    LOG.info(f"Evaluated {len(predictions)} proteins")

    results = evaluate_dataset(predictions, ground_truths, tolerance=args.tolerance)
    simple_avg_iou = results.get('simple_avg_iou', 0.0)
    simple_avg_mcc = results.get('simple_avg_mcc', 0.0)

    print(f"\n{'=' * 60}")
    print(f"EguchiCNN Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Number of proteins:      {results['num_proteins']}")
    print(f"  Boundary tolerance:      +/-{args.tolerance} residues")
    print(f"{'=' * 60}")
    print(f"  Domain-length weighted:")
    print(f"    IoU:                   {results['avg_iou']:.6f}")
    print(f"    Boundary MCC:          {results['avg_mcc']:.6f}")
    print(f"  Simple average:")
    print(f"    IoU:                   {simple_avg_iou:.6f}")
    print(f"    Boundary MCC:          {simple_avg_mcc:.6f}")
    print(f"{'=' * 60}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write("method,num_proteins,avg_iou,avg_mcc,simple_avg_iou,simple_avg_mcc\n")
            f.write(f"eguchi_cnn,{results['num_proteins']},{results['avg_iou']:.4f},{results['avg_mcc']:.4f},{simple_avg_iou:.4f},{simple_avg_mcc:.4f}\n")
        print(f"Results saved to {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate EguchiCNN from HDF5 features')
    parser.add_argument('--test_csv', type=str, default='data/baseline_csvs/test.csv')
    parser.add_argument('--hdf5_path', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='baselines/results/eguchi_cnn_results.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tolerance', type=int, default=20)
    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
