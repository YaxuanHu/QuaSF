"""
SWORD: Parameter tuning on train set + evaluation on test set.

Since SWORD is unsupervised, we tune hyperparameters (contact_threshold,
min_segment_len) on the training set, then evaluate with best params on test set.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import sword_predict
from evaluate_metrics import evaluate_dataset


def parse_domain_regions(domain_regions_str, resi):
    """Parse domain_regions string into per-residue domain labels."""
    resi = np.array(resi)
    N = len(resi)
    domain_label = np.zeros(N, dtype=np.int64)

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
            if '-' in seg:
                parts = seg.split('-')
                start = int(parts[0])
                end = int(parts[1])
                mask = (resi >= start) & (resi <= end)
                domain_label[mask] = dom_idx + 1

    return domain_label


def find_pdb(structures_dir, name):
    if not name or name == 'nan':
        return None
    for n in [name, name.upper(), name.lower()]:
        path = os.path.join(structures_dir, f"{n}.pdb")
        if os.path.exists(path):
            return path
    return None


def run_sword_on_dataset(csv_path, structures_dir, contact_threshold, min_segment_len,
                         max_domains=10, tolerance=20, max_samples=None):
    """Run SWORD on a dataset and return metrics."""
    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.head(max_samples)

    predictions = []
    ground_truths = []
    skipped = 0

    for idx, row in df.iterrows():
        pdb_id = str(row['id']).strip()
        uniprot_id = str(row.get('uniprot', '')).strip()
        domain_regions = str(row.get('domain_regions', ''))

        if pd.isna(domain_regions) or domain_regions == 'nan':
            domain_regions = ''

        pdb_path = find_pdb(structures_dir, uniprot_id) or find_pdb(structures_dir, pdb_id)
        if pdb_path is None:
            skipped += 1
            continue

        pred_labels, pred_resi, n_dom = sword_predict(
            pdb_path,
            contact_threshold=contact_threshold,
            n_domains=None,
            max_domains=max_domains,
            min_segment_len=min_segment_len,
        )

        if pred_labels is None:
            skipped += 1
            continue

        gt_labels = parse_domain_regions(domain_regions, pred_resi)
        if gt_labels.max() == 0:
            skipped += 1
            continue

        ground_truths.append((pdb_id, gt_labels))
        predictions.append((pdb_id, pred_labels))

    if len(predictions) == 0:
        return {'avg_iou': 0.0, 'avg_mcc': 0.0, 'num_proteins': 0}

    results = evaluate_dataset(predictions, ground_truths, tolerance=tolerance)
    return results


def tune_parameters(args):
    """Grid search over hyperparameters on training set."""
    contact_thresholds = [7.0, 8.0, 9.0, 10.0]
    min_segment_lens = [5, 10, 15, 20]

    print("=" * 60)
    print("SWORD Parameter Tuning on Training Set")
    print("=" * 60)

    best_score = -1
    best_params = {}
    all_results = []

    for ct, msl in product(contact_thresholds, min_segment_lens):
        print(f"\nTesting: contact_threshold={ct}, min_segment_len={msl}")

        results = run_sword_on_dataset(
            args.train_csv, args.structures_dir,
            contact_threshold=ct, min_segment_len=msl,
            max_domains=args.max_domains, tolerance=args.tolerance,
            max_samples=args.tune_samples,
        )

        score = results['avg_iou'] + results['avg_mcc']
        print(f"  IoU={results['avg_iou']:.4f}, MCC={results['avg_mcc']:.4f}, "
              f"n={results['num_proteins']}, combined={score:.4f}")

        all_results.append({
            'contact_threshold': ct,
            'min_segment_len': msl,
            'avg_iou': results['avg_iou'],
            'avg_mcc': results['avg_mcc'],
            'num_proteins': results['num_proteins'],
            'combined': score,
        })

        if score > best_score:
            best_score = score
            best_params = {'contact_threshold': ct, 'min_segment_len': msl}

    print(f"\n{'=' * 60}")
    print(f"Best params: {best_params} (combined={best_score:.4f})")
    print(f"{'=' * 60}")

    # Save tuning results
    tune_output = args.output.replace('.csv', '_tuning.csv')
    os.makedirs(os.path.dirname(tune_output) if os.path.dirname(tune_output) else '.', exist_ok=True)
    pd.DataFrame(all_results).to_csv(tune_output, index=False)
    print(f"Tuning results saved to {tune_output}")

    return best_params


def main():
    parser = argparse.ArgumentParser(description='SWORD: tune + evaluate')
    parser.add_argument('--train_csv', type=str, default='data/HeMeNet/train_dataset_with_structure.csv')
    parser.add_argument('--test_csv', type=str, default='data/HeMeNet/test_dataset_with_structure.csv')
    parser.add_argument('--structures_dir', type=str, default='data/HeMeNet/structures')
    parser.add_argument('--output', type=str, default='baselines/results/sword_results.csv')
    parser.add_argument('--max_domains', type=int, default=10)
    parser.add_argument('--tolerance', type=int, default=20)
    parser.add_argument('--tune_samples', type=int, default=500,
                        help='Max samples for tuning (None=all)')
    parser.add_argument('--skip_tune', action='store_true',
                        help='Skip tuning, use default params')
    parser.add_argument('--contact_threshold', type=float, default=8.0)
    parser.add_argument('--min_segment_len', type=int, default=10)
    args = parser.parse_args()

    if not args.skip_tune:
        best_params = tune_parameters(args)
        ct = best_params['contact_threshold']
        msl = best_params['min_segment_len']
    else:
        ct = args.contact_threshold
        msl = args.min_segment_len

    # Evaluate on test set with best params
    print(f"\n{'=' * 60}")
    print(f"Evaluating on test set with contact_threshold={ct}, min_segment_len={msl}")
    print(f"{'=' * 60}")

    results = run_sword_on_dataset(
        args.test_csv, args.structures_dir,
        contact_threshold=ct, min_segment_len=msl,
        max_domains=args.max_domains, tolerance=args.tolerance,
    )

    print(f"\n{'=' * 60}")
    print(f"SWORD Test Results")
    print(f"{'=' * 60}")
    print(f"Number of proteins: {results['num_proteins']}")
    print(f"Average IoU (weighted): {results['avg_iou']:.4f}")
    print(f"Average MCC (weighted): {results['avg_mcc']:.4f}")
    print(f"Simple Average IoU: {results['simple_avg_iou']:.4f}")
    print(f"Simple Average MCC: {results['simple_avg_mcc']:.4f}")
    print(f"{'=' * 60}")

    # Save final results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("method,num_proteins,avg_iou,avg_mcc,simple_avg_iou,simple_avg_mcc,contact_threshold,min_segment_len\n")
        f.write(f"sword,{results['num_proteins']},{results['avg_iou']:.4f},"
                f"{results['avg_mcc']:.4f},{results['simple_avg_iou']:.4f},{results['simple_avg_mcc']:.4f},{ct},{msl}\n")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
