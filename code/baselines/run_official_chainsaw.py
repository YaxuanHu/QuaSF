"""Run official Chainsaw pretrained weights on our test set.

Uses the official model from JudeWells/chainsaw saved_models/model_v3.
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chainsaw-main'))
from get_predictions import load_model, predict

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_metrics import evaluate_single_protein, evaluate_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
LOG = logging.getLogger(__name__)


def parse_chopping(chopping_str, n_res):
    """Parse Chainsaw chopping string into per-residue domain labels.

    Format examples:
        "3-126,135-267" -> 2 domains
        "1-100_200-300,101-199" -> discontinuous domain
    Returns label array [n_res] with 0=unassigned, 1,2,...=domain IDs.
    """
    labels = np.zeros(n_res, dtype=np.int64)
    if not chopping_str or chopping_str == 'NULL':
        return labels

    domains = chopping_str.split(',')
    dom_id = 1
    for dom_str in domains:
        dom_str = dom_str.strip()
        if not dom_str:
            continue
        segments = dom_str.split('_')
        for seg in segments:
            seg = seg.strip()
            if '-' not in seg:
                continue
            parts = seg.split('-')
            start = int(parts[0]) - 1  # 1-indexed to 0-indexed
            end = int(parts[1])
            labels[max(0, start):min(n_res, end)] = dom_id
        dom_id += 1
    return labels


def parse_gt_domain_regions(domain_regions_str, n_res):
    """Parse ground truth domain_regions string into per-residue labels."""
    resi = np.arange(1, n_res + 1)
    domain_label = np.zeros(n_res, dtype=np.int64)

    if not domain_regions_str or domain_regions_str == 'nan' or pd.isna(domain_regions_str):
        domain_label[:] = 1
        return domain_label

    domain_regions_str = str(domain_regions_str).strip()
    if not domain_regions_str:
        domain_label[:] = 1
        return domain_label

    domains = domain_regions_str.split(';')
    for dom_idx, dom_str in enumerate(domains):
        dom_str = dom_str.strip()
        if not dom_str:
            continue
        segments = dom_str.split(',')
        for seg in segments:
            seg = seg.strip()
            if '-' not in seg:
                continue
            parts = seg.split('-')
            start = int(parts[0])
            end = int(parts[1])
            mask = (resi >= start) & (resi <= end)
            domain_label[mask] = dom_idx + 1
    return domain_label


def run_chainsaw(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info(f'Loading Chainsaw official model from {args.model_dir} on {device}')
    model = load_model(
        model_dir=args.model_dir,
        remove_disordered_domain_threshold=args.remove_disordered_domain_threshold,
        min_ss_components=args.min_ss_components,
        min_domain_length=args.min_domain_length,
        post_process_domains=args.post_process_domains,
    )

    df = pd.read_csv(args.test_csv)
    predictions = []
    ground_truths = []
    skipped = 0
    failed = []

    structures_dir = args.structures_dir
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Chainsaw official'):
        pdb_id = str(row['id']).strip()
        domain_regions = str(row.get('domain_regions', ''))

        if pd.isna(domain_regions) or domain_regions == 'nan':
            domain_regions = ''

        pdb_path = os.path.join(structures_dir, f'{pdb_id}.pdb')
        if not os.path.exists(pdb_path):
            skipped += 1
            failed.append((pdb_id, 'no_pdb'))
            continue

        try:
            result = predict(model, pdb_path, renumber_pdbs=args.renumber_pdbs, pdbchain=None)
        except Exception as e:
            LOG.warning(f'Failed to predict {pdb_id}: {e}')
            skipped += 1
            failed.append((pdb_id, str(e)))
            continue

        n_res = result.nres
        pred_labels = parse_chopping(result.chopping, n_res)
        gt_labels = parse_gt_domain_regions(domain_regions, n_res)

        if gt_labels.max() == 0:
            skipped += 1
            failed.append((pdb_id, 'no_gt_domains'))
            continue

        predictions.append((pdb_id, pred_labels))
        ground_truths.append((pdb_id, gt_labels))

    LOG.info(f'Evaluated {len(predictions)} proteins, skipped {skipped}')

    results = evaluate_dataset(predictions, ground_truths, tolerance=args.tolerance)

    print(f"\n{'=' * 60}")
    print(f'Chainsaw Official Results (Zero-shot)')
    print(f"{'=' * 60}")
    print(f'  Number of proteins:      {results["num_proteins"]}')
    print(f'  Boundary tolerance:      ±{args.tolerance} residues')
    print(f"{'=' * 60}")
    print(f'  Domain-length weighted:')
    print(f'    IoU:                   {results["avg_iou"]:.6f}')
    print(f'    Boundary MCC:          {results["avg_mcc"]:.6f}')
    print(f'  Simple average:')
    print(f'    IoU:                   {results["simple_avg_iou"]:.6f}')
    print(f'    Boundary MCC:          {results["simple_avg_mcc"]:.6f}')
    print(f"{'=' * 60}")

    os.makedirs(os.path.dirname(args.output) if args.output else 'code/baselines/results', exist_ok=True)
    if args.output:
        with open(args.output, 'w') as f:
            f.write('method,num_proteins,avg_iou,avg_mcc,simple_avg_iou,simple_avg_mcc\n')
            f.write(f'chainsaw_official,{results["num_proteins"]},'
                    f'{results["avg_iou"]:.4f},{results["avg_mcc"]:.4f},'
                    f'{results["simple_avg_iou"]:.4f},{results["simple_avg_mcc"]:.4f}\n')
        LOG.info(f'Results saved to {args.output}')

    if args.failed_log:
        with open(args.failed_log, 'w') as f:
            for pdb_id, reason in failed:
                f.write(f'{pdb_id}\t{reason}\n')
        LOG.info(f'Failed log saved to {args.failed_log}')

    return results


def main():
    parser = argparse.ArgumentParser(description='Run official Chainsaw on test set')
    parser.add_argument('--test_csv', type=str,
                        default='data/baseline_csvs/test.csv')
    parser.add_argument('--structures_dir', type=str,
                        default='data/dataset_v1/structures_pdb')
    parser.add_argument('--model_dir', type=str,
                        default='code/baselines/chainsaw-main/saved_models/model_v3')
    parser.add_argument('--output', type=str,
                        default='code/baselines/results/chainsaw_official_results.csv')
    parser.add_argument('--failed_log', type=str,
                        default='code/baselines/results/chainsaw_official_failed.txt')
    parser.add_argument('--tolerance', type=int, default=20)
    parser.add_argument('--renumber_pdbs', action='store_true',
                        help='Renumber PDBs before STRIDE (slower, may help with residue numbering)')
    parser.add_argument('--min_domain_length', type=int, default=30)
    parser.add_argument('--min_ss_components', type=int, default=2)
    parser.add_argument('--remove_disordered_domain_threshold', type=float, default=0.35)
    parser.add_argument('--post_process_domains', action='store_true', default=True)
    args = parser.parse_args()
    run_chainsaw(args)


if __name__ == '__main__':
    main()
