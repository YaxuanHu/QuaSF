"""Run official Merizo pretrained weights on our test set.

Uses the official model from psipred/Merizo weights/.
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_metrics import evaluate_single_protein, evaluate_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
LOG = logging.getLogger(__name__)


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


def run_merizo(args):
    # Import here so that sys.path is set up before loading model package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'merizo'))
    from predict import read_split_weight_files, segment
    from model.network import Merizo as _Merizo

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    LOG.info(f'Loading Merizo official weights from {args.weights_dir} on {device}')
    weights = read_split_weight_files(args.weights_dir)
    network = _Merizo()
    network.load_state_dict(weights)
    network = network.to(device)
    network.eval()
    LOG.info(f'Network loaded ({len(weights)} weight tensors)')

    df = pd.read_csv(args.test_csv)
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(df)
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    LOG.info(f'Processing rows {start_idx}:{end_idx} ({len(df)} proteins)')

    predictions = []
    ground_truths = []
    skipped = 0
    failed = []

    structures_dir = args.structures_dir
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Merizo official'):
        pdb_id = str(row['id']).strip()
        domain_regions = str(row.get('domain_regions', ''))

        if pd.isna(domain_regions) or domain_regions == 'nan':
            domain_regions = ''

        pdb_path = os.path.join(structures_dir, f'{pdb_id}.pdb')
        if not os.path.exists(pdb_path):
            skipped += 1
            failed.append((pdb_id, 'no_pdb'))
            continue

        result = None
        try:
            result = segment(
                pdb_path,
                network,
                device,
                length_conditional_iterate=args.length_conditional_iterate,
                iterate=args.iterate,
                max_iterations=args.max_iterations,
                shuffle_indices=False,
                pdb_chain='A',
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and not args.cpu:
                LOG.warning(f'OOM for {pdb_id}, retrying on CPU')
                torch.cuda.empty_cache()
                try:
                    # Create a fresh CPU copy instead of moving the active GPU model
                    network_cpu = _Merizo()
                    network_cpu.load_state_dict(network.state_dict())
                    network_cpu = network_cpu.to('cpu')
                    network_cpu.eval()
                    result = segment(
                        pdb_path,
                        network_cpu,
                        'cpu',
                        length_conditional_iterate=False,
                        iterate=False,
                        max_iterations=args.max_iterations,
                        shuffle_indices=False,
                        pdb_chain='A',
                    )
                    del network_cpu
                    torch.cuda.empty_cache()
                except Exception as e2:
                    LOG.warning(f'CPU retry also failed for {pdb_id}: {e2}')
                    skipped += 1
                    failed.append((pdb_id, f'oom_cpu_fail:{e2}'))
                    torch.cuda.empty_cache()
                    continue
            else:
                LOG.warning(f'Failed to predict {pdb_id}: {e}')
                skipped += 1
                failed.append((pdb_id, str(e)))
                continue
        except Exception as e:
            LOG.warning(f'Failed to predict {pdb_id}: {e}')
            skipped += 1
            failed.append((pdb_id, str(e)))
            continue

        n_res = result['nres']
        pred_labels = result['domain_ids'].cpu().numpy().astype(np.int64).flatten()[:n_res]
        gt_labels = parse_gt_domain_regions(domain_regions, n_res)

        if gt_labels.max() == 0:
            skipped += 1
            failed.append((pdb_id, 'no_gt_domains'))
            continue

        predictions.append((pdb_id, pred_labels))
        ground_truths.append((pdb_id, gt_labels))

    LOG.info(f'Evaluated {len(predictions)} proteins, skipped {skipped}')

    results = evaluate_dataset(predictions, ground_truths, tolerance=args.tolerance)

    # Write per-protein metrics so multiple chunks can be merged exactly.
    gt_dict = {pid: labels for pid, labels in ground_truths}
    per_protein_path = args.output.replace('.csv', '_per_protein.csv') if args.output else None
    if per_protein_path:
        with open(per_protein_path, 'w') as f:
            f.write('id,length,iou,mcc,weight\n')
            for pdb_id, metrics in results['per_protein'].items():
                gt_labels = gt_dict[pdb_id]
                weight = int(max(np.sum(np.array(gt_labels) > 0), 1))
                f.write(f'{pdb_id},{len(gt_labels)},{metrics["iou"]:.6f},'
                        f'{metrics["mcc"]:.6f},{weight}\n')
        LOG.info(f'Per-protein metrics saved to {per_protein_path}')

    print(f"\n{'=' * 60}")
    print(f'Merizo Official Results (Zero-shot)')
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
            f.write(f'merizo_official,{results["num_proteins"]},'
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
    parser = argparse.ArgumentParser(description='Run official Merizo on test set')
    parser.add_argument('--test_csv', type=str,
                        default='data/baseline_csvs/test.csv')
    parser.add_argument('--structures_dir', type=str,
                        default='data/dataset_v1/structures_pdb')
    parser.add_argument('--weights_dir', type=str,
                        default='code/baselines/merizo/weights')
    parser.add_argument('--output', type=str,
                        default='code/baselines/results/merizo_official_results.csv')
    parser.add_argument('--failed_log', type=str,
                        default='code/baselines/results/merizo_official_failed.txt')
    parser.add_argument('--tolerance', type=int, default=20)
    parser.add_argument('--iterate', action='store_true', default=False,
                        help='Enable iterative segmentation (slower, more memory)')
    parser.add_argument('--length_conditional_iterate', action='store_true', default=False,
                        help='Auto-enable iterate for proteins >512 residues')
    parser.add_argument('--max_iterations', type=int, default=10)
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Start row index (inclusive) in test_csv')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End row index (exclusive) in test_csv')
    args = parser.parse_args()
    run_merizo(args)


if __name__ == '__main__':
    main()
