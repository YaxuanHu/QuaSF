"""Evaluation script for SWORD on test set."""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import sword_predict, extract_ca_coords
from evaluate_metrics import evaluate_single_protein, evaluate_dataset


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
    """Try to find PDB file with various naming conventions."""
    if not name or name == 'nan':
        return None
    for n in [name, name.upper(), name.lower()]:
        path = os.path.join(structures_dir, f"{n}.pdb")
        if os.path.exists(path):
            return path
    return None


def evaluate(args):
    print(f"SWORD Evaluation")
    print(f"Contact threshold: {args.contact_threshold} Å")
    print(f"Min segment length: {args.min_segment_len}")
    print(f"Max domains: {args.max_domains}")

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

        # Run SWORD prediction
        pred_labels, pred_resi, n_dom = sword_predict(
            pdb_path,
            contact_threshold=args.contact_threshold,
            n_domains=None,
            max_domains=args.max_domains,
            min_segment_len=args.min_segment_len,
        )

        if pred_labels is None:
            skipped += 1
            continue

        # Get GT labels using the same residue indices
        gt_labels = parse_domain_regions(domain_regions, pred_resi)

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

    # Compute simple (unweighted) averages from per-protein results
    per_protein = results.get('per_protein', {})
    if len(per_protein) > 0:
        simple_avg_iou = np.mean([m['iou'] for m in per_protein.values()])
        simple_avg_mcc = np.mean([m['mcc'] for m in per_protein.values()])
    else:
        simple_avg_iou = 0.0
        simple_avg_mcc = 0.0

    print(f"\n{'=' * 60}")
    print(f"SWORD Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Number of proteins:      {results['num_proteins']}")
    print(f"  Boundary tolerance:      ±{args.tolerance} residues")
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
            f.write(f"sword,{results['num_proteins']},{results['avg_iou']:.4f},{results['avg_mcc']:.4f},{simple_avg_iou:.4f},{simple_avg_mcc:.4f}\n")
        print(f"Results saved to {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SWORD domain segmentation')
    parser.add_argument('--test_csv', type=str, default='data/HeMeNet/test_dataset_with_structure.csv')
    parser.add_argument('--structures_dir', type=str, default='data/HeMeNet/structures')
    parser.add_argument('--output', type=str, default='baselines/results/sword_results.csv')
    parser.add_argument('--contact_threshold', type=float, default=8.0)
    parser.add_argument('--max_domains', type=int, default=10)
    parser.add_argument('--min_segment_len', type=int, default=10)
    parser.add_argument('--tolerance', type=int, default=20)
    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
