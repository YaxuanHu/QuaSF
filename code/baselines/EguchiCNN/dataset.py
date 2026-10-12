"""Dataset for EguchiCNN: extracts Ca distance matrices from PDB files.

The paper uses Ca pairwise distance matrices as input, scaled by /100,
zero-padded to max_len (512) for shorter proteins, center-cropped for longer ones.
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
import h5py

LOG = logging.getLogger(__name__)

_pdb_parser = PDBParser(QUIET=True)


def extract_ca_coords(pdb_path, chain_id='A'):
    """Extract Ca coordinates from a PDB file.

    Args:
        pdb_path: path to PDB file
        chain_id: chain to extract (tries first available if not found)

    Returns:
        coords: np.array [N, 3] Ca coordinates, or None on failure
    """
    try:
        structure = _pdb_parser.get_structure('protein', pdb_path)
    except Exception as e:
        LOG.warning(f"Failed to parse PDB {pdb_path}: {e}")
        return None

    model = structure[0]
    chains = list(model.get_chains())
    if not chains:
        return None

    # Find target chain
    chain = None
    for c in chains:
        if c.id == chain_id:
            chain = c
            break
    if chain is None:
        chain = chains[0]

    coords = []
    for residue in chain.get_residues():
        # Skip hetero residues
        if residue.id[0] != ' ':
            continue
        if 'CA' in residue:
            coords.append(residue['CA'].get_vector().get_array())

    if len(coords) == 0:
        return None

    return np.array(coords, dtype=np.float32)


def compute_distance_matrix(coords):
    """Compute pairwise Ca distance matrix.

    Args:
        coords: np.array [N, 3]

    Returns:
        dist_matrix: np.array [N, N]
    """
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist.astype(np.float32)


def pad_or_crop(matrix, labels, max_len):
    """Pad (zero) or center-crop distance matrix and labels to max_len.

    Args:
        matrix: np.array [N, N]
        labels: np.array [N]
        max_len: target size

    Returns:
        matrix_out: np.array [max_len, max_len]
        labels_out: np.array [max_len]
        mask: np.array [max_len] bool, True for valid residues
    """
    N = matrix.shape[0]

    if N <= max_len:
        # Zero-pad
        mat_out = np.zeros((max_len, max_len), dtype=np.float32)
        mat_out[:N, :N] = matrix
        lab_out = np.zeros(max_len, dtype=np.int64)
        lab_out[:N] = labels
        mask = np.zeros(max_len, dtype=bool)
        mask[:N] = True
    else:
        # Center-crop
        start = (N - max_len) // 2
        end = start + max_len
        mat_out = matrix[start:end, start:end].copy()
        lab_out = labels[start:end].copy()
        mask = np.ones(max_len, dtype=bool)

    return mat_out, lab_out, mask


def parse_domain_regions(domain_regions_str, n_res):
    """Parse domain_regions string into per-residue domain labels.

    Args:
        domain_regions_str: e.g. "45-294" or "37-216;313-437;443-564"
        n_res: number of residues

    Returns:
        labels: np.array [n_res], 0=linker/bg, 1..K=domain IDs
    """
    resi = np.arange(1, n_res + 1)
    labels = np.zeros(n_res, dtype=np.int64)

    if not domain_regions_str or domain_regions_str == 'nan' or pd.isna(domain_regions_str):
        labels[:] = 1
        return labels

    domain_regions_str = str(domain_regions_str).strip()
    if not domain_regions_str:
        labels[:] = 1
        return labels

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
                labels[mask] = dom_idx + 1

    return labels


def find_pdb(structures_dir, name):
    """Try to find PDB file with various naming conventions."""
    if not name or name == 'nan':
        return None
    for n in [name, name.upper(), name.lower()]:
        path = os.path.join(structures_dir, f"{n}.pdb")
        if os.path.exists(path):
            return path
    return None


class EguchiCNNH5Dataset(Dataset):
    """Dataset that reads precomputed distance matrices from an HDF5 file."""

    def __init__(self, csv_path, hdf5_path, max_len=512):
        self.hdf5_path = hdf5_path
        self.max_len = max_len

        df = pd.read_csv(csv_path)
        self.samples = []
        for _, row in df.iterrows():
            pdb_id = str(row['id']).strip()
            uniprot_id = str(row.get('uniprot', '')).strip()
            domain_regions = row.get('domain_regions', '')
            if pd.isna(domain_regions):
                domain_regions = ''
            self.samples.append({
                'pdb_id': pdb_id,
                'uniprot': uniprot_id,
                'domain_regions': str(domain_regions),
            })

        self.h5 = None
        self.index = None
        LOG.info(f"EguchiCNNH5Dataset: {len(self.samples)} rows from {csv_path}")

    def _lazy_open(self):
        if self.h5 is not None:
            return
        self.h5 = h5py.File(self.hdf5_path, 'r')
        pdb_ids = self.h5['pdb_id'][:]
        # Map pdb_id string -> HDF5 row index
        self.index = {pid.decode() if isinstance(pid, bytes) else pid: i for i, pid in enumerate(pdb_ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        self._lazy_open()

        key = info['pdb_id']
        if key not in self.index:
            LOG.warning(f"Missing HDF5 entry for {key}")
            # Return a dummy sample to keep DDP collective ops stable
            return {
                'pdb_id': key,
                'input': torch.zeros(1, self.max_len, self.max_len, dtype=torch.float32),
                'labels': torch.zeros(self.max_len, dtype=torch.long),
                'mask': torch.zeros(self.max_len, dtype=torch.bool),
                'n_res': 0,
            }

        h5_idx = self.index[key]
        return {
            'pdb_id': key,
            'input': torch.from_numpy(self.h5['dist_matrix'][h5_idx]).float(),  # [1, L, L]
            'labels': torch.from_numpy(self.h5['labels'][h5_idx]).long(),       # [L]
            'mask': torch.from_numpy(self.h5['mask'][h5_idx]).bool(),           # [L]
            'n_res': int(self.h5['n_res'][h5_idx]),
        }


class EguchiCNNDataset(Dataset):
    """Dataset that loads cached distance matrices or computes them from PDB files."""

    def __init__(self, csv_path, structures_dir, max_len=512, cache_dir=None):
        self.structures_dir = structures_dir
        self.max_len = max_len
        self.cache_dir = cache_dir

        df = pd.read_csv(csv_path)
        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            pdb_id = str(row['id']).strip()
            uniprot_id = str(row.get('uniprot', '')).strip()
            domain_regions = row.get('domain_regions', '')

            pdb_path = find_pdb(structures_dir, uniprot_id) or find_pdb(structures_dir, pdb_id)
            if pdb_path is None:
                skipped += 1
                continue

            cache_key = os.path.splitext(os.path.basename(pdb_path))[0]
            cache_path = os.path.join(cache_dir, f"{cache_key}.pt") if cache_dir else None

            self.samples.append({
                'pdb_id': pdb_id,
                'uniprot': uniprot_id,
                'pdb_path': pdb_path,
                'domain_regions': str(domain_regions) if not pd.isna(domain_regions) else '',
                'cache_path': cache_path,
            })

        LOG.info(f"EguchiCNNDataset: {len(self.samples)} samples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]

        # Try cache first
        if info.get('cache_path') and os.path.exists(info['cache_path']):
            try:
                cached = torch.load(info['cache_path'], map_location='cpu', weights_only=True)
                return {
                    'pdb_id': info['pdb_id'],
                    'input': cached['dist_matrix'].unsqueeze(0),  # [1, max_len, max_len]
                    'labels': cached['labels'],                    # [max_len]
                    'mask': cached['mask'],                        # [max_len]
                    'n_res': cached['n_res'],
                }
            except Exception as e:
                LOG.warning(f"Failed to load cache {info['cache_path']}: {e}")

        # Extract Ca coordinates
        coords = extract_ca_coords(info['pdb_path'])
        if coords is None or len(coords) < 10:
            # Retry with a random valid sample to avoid None in DDP
            return self.__getitem__(np.random.randint(len(self)))

        n_res = len(coords)

        # Compute distance matrix
        dist_matrix = compute_distance_matrix(coords)

        # Scale by /100 as per paper
        dist_matrix = dist_matrix / 100.0

        # Parse domain labels
        labels = parse_domain_regions(info['domain_regions'], n_res)

        # Pad or crop
        dist_matrix, labels, mask = pad_or_crop(dist_matrix, labels, self.max_len)

        # To tensors: [1, max_len, max_len]
        dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32).unsqueeze(0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        return {
            'pdb_id': info['pdb_id'],
            'input': dist_tensor,        # [1, max_len, max_len]
            'labels': labels_tensor,      # [max_len]
            'mask': mask_tensor,          # [max_len]
            'n_res': n_res,
        }


def collate_fn(batch):
    """Collate that filters None and stacks tensors."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        # Return a dummy batch instead of None to keep DDP collective ops in sync
        LOG.warning("collate_fn: entire batch is None, returning dummy batch")
        dummy_len = 512
        return {
            'pdb_id': ['dummy'],
            'input': torch.zeros(1, 1, dummy_len, dummy_len, dtype=torch.float32),
            'labels': torch.zeros(1, dummy_len, dtype=torch.long),
            'mask': torch.zeros(1, dummy_len, dtype=torch.bool),
            'n_res': [0],
        }

    return {
        'pdb_id': [b['pdb_id'] for b in batch],
        'input': torch.stack([b['input'] for b in batch]),      # [B, 1, L, L]
        'labels': torch.stack([b['labels'] for b in batch]),    # [B, L]
        'mask': torch.stack([b['mask'] for b in batch]),        # [B, L]
        'n_res': [b['n_res'] for b in batch],
    }
