"""Fast multiprocessing conversion of EguchiCNN .pt cache files to HDF5.

Avoids gzip compression (slow) and reads .pt files in parallel, writing
results in input order so no large in-memory buffer is needed.
"""
import os
import sys
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch
import h5py

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
LOG = logging.getLogger(__name__)


def build_lookup(dir_path, ext):
    """Build {basename: full_path} lookup for files in dir_path."""
    LOG.info(f"Building lookup for {dir_path}...")
    lookup = {}
    for name in os.listdir(dir_path):
        if name.endswith(ext):
            lookup[name[:-len(ext)]] = os.path.join(dir_path, name)
    LOG.info(f"  {len(lookup)} {ext} files found")
    return lookup


def load_one(sample):
    """Load a single .pt cache file, return dict of numpy arrays."""
    try:
        cached = torch.load(sample['cache_path'], map_location='cpu', weights_only=True)
        return {
            'pdb_id': sample['pdb_id'],
            'dist_matrix': cached['dist_matrix'].numpy(),  # [1, L, L]
            'labels': cached['labels'].numpy(),            # [L]
            'mask': cached['mask'].numpy(),                # [L]
            'n_res': int(cached['n_res']),
        }
    except Exception as e:
        return {'error': str(e), 'cache_path': sample['cache_path']}


def convert_split(csv_path, structures_lookup, cache_lookup, hdf5_path, max_len, num_workers=32, map_chunksize=64):
    """Convert one CSV/split's cache files to HDF5."""
    df = pd.read_csv(csv_path)

    samples = []
    for _, row in df.iterrows():
        pdb_id = str(row['id']).strip()
        uniprot_id = str(row.get('uniprot', '')).strip()

        pdb_path = structures_lookup.get(uniprot_id) or structures_lookup.get(pdb_id)
        if pdb_path is None:
            continue

        cache_key = os.path.splitext(os.path.basename(pdb_path))[0]
        cache_path = cache_lookup.get(cache_key)
        if cache_path is None:
            continue

        samples.append({
            'pdb_id': pdb_id,
            'cache_path': cache_path,
        })

    n = len(samples)
    LOG.info(f"Converting {n} samples for {csv_path} -> {hdf5_path}")

    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, 'w') as f:
        # No compression for fast writes/reads; chunk for partial reads
        dist_ds = f.create_dataset(
            'dist_matrix', shape=(n, 1, max_len, max_len), dtype='float32',
            chunks=(1, 1, max_len, max_len),
        )
        labels_ds = f.create_dataset(
            'labels', shape=(n, max_len), dtype='int64',
            chunks=(map_chunksize, max_len),
        )
        mask_ds = f.create_dataset(
            'mask', shape=(n, max_len), dtype='bool',
            chunks=(map_chunksize, max_len),
        )
        nres_ds = f.create_dataset('n_res', shape=(n,), dtype='int32')
        pdb_ds = f.create_dataset('pdb_id', shape=(n,), dtype=h5py.string_dtype())

        written = 0
        errors = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for data in executor.map(load_one, samples, chunksize=map_chunksize):
                if 'error' in data:
                    LOG.warning(f"Failed to load {data['cache_path']}: {data['error']}")
                    errors += 1
                    # Leave a blank row to keep indices aligned
                    dist_ds[written] = 0
                    labels_ds[written] = 0
                    mask_ds[written] = False
                    nres_ds[written] = 0
                    pdb_ds[written] = ''
                    written += 1
                    continue

                dist_ds[written] = data['dist_matrix']
                labels_ds[written] = data['labels']
                mask_ds[written] = data['mask']
                nres_ds[written] = data['n_res']
                pdb_ds[written] = data['pdb_id']
                written += 1

                if written % 1000 == 0:
                    LOG.info(f"  {written}/{n} written")

        LOG.info(f"Done: {hdf5_path} ({written} written, {errors} errors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, action='append', required=True)
    parser.add_argument('--structures_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--map_chunksize', type=int, default=64)
    args = parser.parse_args()

    structures_lookup = build_lookup(args.structures_dir, '.pdb')
    cache_lookup = build_lookup(args.cache_dir, '.pt')

    for csv_path in args.csv:
        split_name = os.path.splitext(os.path.basename(csv_path))[0]
        hdf5_path = os.path.join(args.output_dir, f"{split_name}.h5")
        convert_split(
            csv_path, structures_lookup, cache_lookup, hdf5_path,
            args.max_len, args.num_workers, args.map_chunksize,
        )


if __name__ == '__main__':
    main()
