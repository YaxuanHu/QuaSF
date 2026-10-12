"""Convert EguchiCNN .pt cache files to a single HDF5 file per split.

This avoids the massive filesystem overhead of 278k individual .pt files.
"""
import os
import sys
import argparse
import logging

import pandas as pd
import torch
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dataset import find_pdb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
LOG = logging.getLogger(__name__)


def convert_split(csv_path, structures_dir, cache_dir, hdf5_path, max_len, chunk_size=1024):
    """Convert one CSV/split's cache files to HDF5."""
    df = pd.read_csv(csv_path)

    samples = []
    for _, row in df.iterrows():
        pdb_id = str(row['id']).strip()
        uniprot_id = str(row.get('uniprot', '')).strip()
        domain_regions = row.get('domain_regions', '')
        if pd.isna(domain_regions):
            domain_regions = ''

        pdb_path = find_pdb(structures_dir, uniprot_id) or find_pdb(structures_dir, pdb_id)
        if pdb_path is None:
            continue

        cache_key = os.path.splitext(os.path.basename(pdb_path))[0]
        cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
        if not os.path.exists(cache_path):
            continue

        samples.append({
            'pdb_id': pdb_id,
            'cache_path': cache_path,
        })

    n = len(samples)
    LOG.info(f"Converting {n} samples for {csv_path} -> {hdf5_path}")

    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, 'w') as f:
        # Create resizable datasets
        dist_ds = f.create_dataset(
            'dist_matrix', shape=(n, 1, max_len, max_len), dtype='float32',
            chunks=(1, 1, max_len, max_len), compression='gzip', compression_opts=1,
        )
        labels_ds = f.create_dataset(
            'labels', shape=(n, max_len), dtype='int64',
            chunks=(chunk_size, max_len), compression='gzip', compression_opts=1,
        )
        mask_ds = f.create_dataset(
            'mask', shape=(n, max_len), dtype='bool',
            chunks=(chunk_size, max_len), compression='gzip', compression_opts=1,
        )
        nres_ds = f.create_dataset('n_res', shape=(n,), dtype='int32')
        pdb_ds = f.create_dataset('pdb_id', shape=(n,), dtype=h5py.string_dtype())

        for i, sample in enumerate(samples):
            try:
                cached = torch.load(sample['cache_path'], map_location='cpu', weights_only=True)
                dist_ds[i] = cached['dist_matrix'].numpy().reshape(1, max_len, max_len)
                labels_ds[i] = cached['labels'].numpy()
                mask_ds[i] = cached['mask'].numpy()
                nres_ds[i] = cached['n_res']
                pdb_ds[i] = sample['pdb_id']
            except Exception as e:
                LOG.warning(f"Failed to load {sample['cache_path']}: {e}")
                continue

            if (i + 1) % 1000 == 0:
                LOG.info(f"  {i+1}/{n} done")

    LOG.info(f"Done: {hdf5_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, action='append', required=True)
    parser.add_argument('--structures_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=512)
    args = parser.parse_args()

    for csv_path in args.csv:
        split_name = os.path.splitext(os.path.basename(csv_path))[0]
        hdf5_path = os.path.join(args.output_dir, f"{split_name}.h5")
        convert_split(csv_path, args.structures_dir, args.cache_dir, hdf5_path, args.max_len)


if __name__ == '__main__':
    main()
