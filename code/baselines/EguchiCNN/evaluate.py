"""Evaluation script for EguchiCNN on test set.

EguchiCNN is a 2D CNN that performs semantic segmentation on protein
Ca distance matrices, predicting per-residue domain labels.

Reference: Eguchi & Huang, Bioinformatics 2020.

Output format aligned with baselines/chainsaw/evaluate.py.
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import EguchiCNN
from dataset import extract_ca_coords, compute_distance_matrix, pad_or_crop, parse_domain_regions, find_pdb
from evaluate_metrics import evaluate_single_protein, evaluate_dataset

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

    # Try Lightning checkpoint first, then raw weights
    ckpt_path = os.path.join(checkpoint_dir, 'best.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, 'last.ckpt')

    if os.path.exists(ckpt_path):
        # Load from Lightning checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Remove 'model.' prefix from Lightning state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        LOG.info(f"Loaded model from {ckpt_path}")
    else:
        # Try raw weights
        weights_path = os.path.join(checkpoint_dir, 'weights.pt')
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            LOG.info(f"Loaded model from {weights_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    model.to(device)
    model.eval()
    return model, config


def predict_single(model, pdb_path, device, max_len=512):
    """Run EguchiCNN prediction on a single PDB file.

    Returns:
        pred_labels: np.array [n_res] per-residue domain labels
        n_res: int
    """
    coords = extract_ca_coords(pdb_path)
    if coords is None or len(coords) < 10:
        return None, 0

    n_res = len(coords)
    dist_matrix = compute_distance_matrix(coords)
    dist_matrix = dist_matrix / 100.0

    # Pad/crop
    dummy_labels = np.zeros(n_res, dtype=np.int64)
    dist_padded, _, mask = pad_or_crop(dist_matrix, dummy_labels, max_len)

    # To tensor
    x = torch.tensor(dist_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    # x: [1, 1, max_len, max_len]

    with torch.no_grad():
        logits = model(x)  # [1, C, max_len]

    preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [max_len]

    # Extract valid predictions
    if n_res <= max_len:
        pred_labels = preds[:n_res]
    else:
        # Was center-cropped, need to reconstruct full-length prediction
        # Assign 0 (linker) to cropped regions
        start = (n_res - max_len) // 2
        pred_labels = np.zeros(n_res, dtype=np.int64)
        pred_labels[start:start + max_len] = preds

    return pred_labels, n_res


def parse_gt_domain_regions(domain_regions_str, n_res):
    """Parse ground truth domain_regions string into per-residue labels."""
    return parse_domain_regions(domain_regions_str, n_res)


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"EguchiCNN Evaluation")
    print(f"Checkpoint: {args.checkpoint_dir}")

    model, config = load_model(args.checkpoint_dir, device)
    max_len = config.get('max_len', 512)

    df = pd.read_csv(args.test_csv)
    predictions = []
    ground_truths = []
    skipped = 0

    for idx, row in df.iterrows():
        pdb_id = str(row['id']).strip()
        uniprot_id = str(row.get('uniprot', '')).strip()
        domain_regions = str(row.get('domain_regions', ''))

        if pd.isna(domain_regions) or domain_regions == 'nan':
            domain_regions = ''

        # Find PDB
        pdb_path = find_pdb(args.structures_dir, uniprot_id) or find_pdb(args.structures_dir, pdb_id)
        if pdb_path is None:
            skipped += 1
            continue

        # Run prediction
        pred_labels, n_res = predict_single(model, pdb_path, device, max_len=max_len)
        if pred_labels is None:
            skipped += 1
            continue

        # Get GT labels
        gt_labels = parse_gt_domain_regions(domain_regions, n_res)

        # Skip if no valid GT domains
        if gt_labels.max() == 0:
            skipped += 1
            continue

        ground_truths.append((pdb_id, gt_labels))
        predictions.append((pdb_id, pred_labels))

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)} proteins...")

    print(f"\nEvaluated {len(predictions)} proteins, skipped {skipped}")

    # Compute metrics
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

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(f"method,num_proteins,avg_iou,avg_mcc,simple_avg_iou,simple_avg_mcc\n")
            f.write(f"eguchi_cnn,{results['num_proteins']},{results['avg_iou']:.4f},{results['avg_mcc']:.4f},{simple_avg_iou:.4f},{simple_avg_mcc:.4f}\n")
        print(f"Results saved to {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate EguchiCNN domain prediction')
    parser.add_argument('--test_csv', type=str,
                        default='data/HeMeNet/test_dataset_with_structure.csv')
    parser.add_argument('--structures_dir', type=str,
                        default='data/HeMeNet/structures')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='baselines/EguchiCNN/checkpoints')
    parser.add_argument('--output', type=str,
                        default='baselines/results/eguchi_cnn_results.csv')
    parser.add_argument('--tolerance', type=int, default=20)
    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
