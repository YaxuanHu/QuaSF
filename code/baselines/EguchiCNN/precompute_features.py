"""Precompute EguchiCNN distance matrices and labels from PDB files.

Saves per-protein .pt files containing:
    dist_matrix: [max_len, max_len] padded/cropped distance matrix
    labels: [max_len] padded/cropped domain labels
    mask: [max_len] valid residue mask
    n_res: int original number of residues
"""
import os
import sys
import argparse
import logging
from multiprocessing import Pool

import pandas as pd
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dataset import extract_ca_coords, compute_distance_matrix, pad_or_crop, parse_domain_regions, find_pdb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
LOG = logging.getLogger(__name__)


def process_one(args):
    """Process a single protein and save cache."""
    pdb_path, cache_path, pdb_id, max_len, domain_regions = args

    if os.path.exists(cache_path):
        return pdb_id, 'skipped'

    try:
        coords = extract_ca_coords(pdb_path)
        if coords is None or len(coords) < 10:
            return pdb_id, 'failed_coords'

        n_res = len(coords)
        dist_matrix = compute_distance_matrix(coords)
        dist_matrix = dist_matrix / 100.0

        labels = parse_domain_regions(domain_regions, n_res)
        dist_padded, labels_padded, mask = pad_or_crop(dist_matrix, labels, max_len)

        torch.save({
            'dist_matrix': torch.tensor(dist_padded, dtype=torch.float32),
            'labels': torch.tensor(labels_padded, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'n_res': n_res,
        }, cache_path)
        return pdb_id, 'ok'
    except Exception as e:
        LOG.warning(f"Failed to process {pdb_id}: {e}")
        return pdb_id, f'error: {e}'


def main():
    parser = argparse.ArgumentParser(description='Precompute EguchiCNN distance matrices')
    parser.add_argument('--csv', type=str, action='append', required=True,
                        help='CSV file(s) to process')
    parser.add_argument('--structures_dir', type=str, required=True,
                        help='Directory containing PDB files')
    parser.add_argument('--cache_dir', type=str, default='code/baselines/EguchiCNN/feature_cache',
                        help='Output directory for cached features')
    parser.add_argument('--max_len', type=int, default=1024,
                        help='Max sequence length / contact map size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    tasks = []
    seen = set()

    for csv_path in args.csv:
        if not os.path.exists(csv_path):
            LOG.warning(f"CSV not found: {csv_path}, skipping")
            continue

        df = pd.read_csv(csv_path)
        LOG.info(f"Processing {csv_path} ({len(df)} rows)")

        for _, row in df.iterrows():
            pdb_id = str(row['id']).strip()
            uniprot_id = str(row.get('uniprot', '')).strip()
            domain_regions = row.get('domain_regions', '')

            if pd.isna(domain_regions):
                domain_regions = ''

            pdb_path = find_pdb(args.structures_dir, uniprot_id) or find_pdb(args.structures_dir, pdb_id)
            if pdb_path is None:
                continue

            cache_key = os.path.splitext(os.path.basename(pdb_path))[0]
            if cache_key in seen:
                continue
            seen.add(cache_key)

            cache_path = os.path.join(args.cache_dir, f"{cache_key}.pt")
            tasks.append((pdb_path, cache_path, cache_key, args.max_len, str(domain_regions)))

    LOG.info(f"Total unique PDB files to process: {len(tasks)}")
    n_cached = sum(1 for _, cp, _, _, _ in tasks if os.path.exists(cp))
    LOG.info(f"Already cached: {n_cached}, remaining: {len(tasks) - n_cached}")

    if n_cached == len(tasks):
        LOG.info("All features already cached. Nothing to do.")
        return

    results = {'ok': 0, 'skipped': 0, 'failed_coords': 0, 'error': 0}

    if args.num_workers <= 1:
        for task in tasks:
            pdb_id, status = process_one(task)
            key = status.split(':')[0].strip()
            results[key] = results.get(key, 0) + 1
            if results['ok'] % 100 == 0 and status == 'ok':
                LOG.info(f"Progress: {sum(results.values())}/{len(tasks)} (ok={results['ok']})")
    else:
        with Pool(args.num_workers) as pool:
            for i, (pdb_id, status) in enumerate(pool.imap_unordered(process_one, tasks)):
                key = status.split(':')[0].strip()
                results[key] = results.get(key, 0) + 1
                if (i + 1) % 100 == 0:
                    LOG.info(f"Progress: {i+1}/{len(tasks)} (ok={results['ok']}, skipped={results['skipped']}, failed={results['failed_coords']})")

    LOG.info(f"Done! Results: {results}")
    LOG.info(f"Cache directory: {args.cache_dir}")


if __name__ == '__main__':
    main()
