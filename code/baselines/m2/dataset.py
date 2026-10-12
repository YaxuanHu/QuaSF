"""
Dataset for end-to-end domain function prediction.
Loads JSONL format dataset with pre-computed ESM features.

ESM feature format: dict with 'residue_emb' key → [L, 1280]
JSONL esm_path is relative to data_root, e.g. "data/dataset_v2/esm_features_v2/XXX.pt"
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict
import os


class EndToEndDomainFunctionDataset(Dataset):
    """
    Each sample is one protein with:
    - Pre-computed ESM residue features (loaded from .pt file)
    - Domain segmentation mask (from domain segments)
    - Function labels: protein-level multi-label (union of all domain labels)
    - Per-domain function labels (for domain-level training)
    """

    def __init__(
        self,
        jsonl_path: str,
        num_classes: int = 256,
        max_len: int = 1024,
        esm_feature_root: str = None,
        data_root: str = "/geniusland/home/huyaxuan/code/region_function_old",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.esm_feature_root = esm_feature_root
        self.data_root = data_root

        # Load all samples
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.samples.append(record)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]

        sequence = record["sequence"]
        seq_len = min(len(sequence), self.max_len)
        domains = record["domains"]

        # ============ Load ESM features ============
        esm_path = record["esm_path"]
        if self.esm_feature_root is not None:
            # Use explicit root + filename
            esm_path = os.path.join(self.esm_feature_root, os.path.basename(esm_path))
        else:
            # esm_path is relative to data_root
            esm_path = os.path.join(self.data_root, esm_path)

        esm_data = torch.load(esm_path, map_location="cpu", weights_only=False)

        # ESM feature format: dict with 'residue_emb' → [L, 1280]
        if isinstance(esm_data, dict):
            if "residue_emb" in esm_data:
                residue_feat = esm_data["residue_emb"]  # [L, 1280]
            elif "residue_features" in esm_data:
                residue_feat = esm_data["residue_features"]
            elif "representations" in esm_data:
                residue_feat = esm_data["representations"]
            else:
                # Try first tensor value
                residue_feat = next(v for v in esm_data.values() if isinstance(v, torch.Tensor) and v.dim() == 2)
        else:
            residue_feat = esm_data  # Tensor directly

        # Truncate to max_len
        feat_len = residue_feat.size(0)
        if feat_len > self.max_len:
            residue_feat = residue_feat[: self.max_len]
            feat_len = self.max_len

        # ============ Build domain mask ============
        # Binary mask: 1 = domain region, 0 = linker/padding
        domain_mask = torch.zeros(self.max_len, dtype=torch.float32)
        for domain in domains:
            for seg in domain["segments"]:
                # 1-based inclusive → 0-based [start, end)
                s = max(0, seg[0] - 1)
                e = min(seq_len, seg[1])
                domain_mask[s:e] = 1.0

        # ============ Build per-domain info ============
        # For each domain: segments, function_label [num_classes], valid flag
        domain_infos = []
        for domain in domains:
            func_label = torch.zeros(self.num_classes, dtype=torch.float32)
            func_valid = domain.get("function_label_valid", False)

            if func_valid:
                for idx_val in domain.get("function_cluster_l2_indices", []):
                    if 0 <= idx_val < self.num_classes:
                        func_label[idx_val] = 1.0

            # Build per-domain binary mask
            d_mask = torch.zeros(self.max_len, dtype=torch.float32)
            for seg in domain["segments"]:
                s = max(0, seg[0] - 1)
                e = min(seq_len, seg[1])
                d_mask[s:e] = 1.0

            domain_infos.append({
                "mask": d_mask,
                "label": func_label,
                "valid": func_valid,
            })

        # ============ Protein-level label (union of all valid domain labels) ============
        protein_label = torch.zeros(self.num_classes, dtype=torch.float32)
        for d_info in domain_infos:
            if d_info["valid"]:
                protein_label = torch.max(protein_label, d_info["label"])

        # ============ Pad ESM features to max_len ============
        pad_len = self.max_len - feat_len
        if pad_len > 0:
            residue_feat = torch.cat(
                [residue_feat, torch.zeros(pad_len, residue_feat.size(-1))], dim=0
            )

        # Attention mask
        attn_mask = torch.zeros(self.max_len, dtype=torch.float32)
        attn_mask[:feat_len] = 1.0

        return {
            "residue_feat": residue_feat,       # [max_len, D]
            "attn_mask": attn_mask,             # [max_len]
            "domain_mask": domain_mask,         # [max_len]
            "protein_label": protein_label,     # [num_classes]
            "domain_infos": domain_infos,       # List[dict] with mask, label, valid
            "seq_len": feat_len,
            "uniprot_acc": record["uniprot_acc"],
        }


def collate_fn(batch: List[Dict]):
    """Custom collate: stack tensors, keep domain_infos as list."""
    return {
        "residue_feat": torch.stack([b["residue_feat"] for b in batch]),   # [B, L, D]
        "attn_mask": torch.stack([b["attn_mask"] for b in batch]),         # [B, L]
        "domain_mask": torch.stack([b["domain_mask"] for b in batch]),     # [B, L]
        "protein_label": torch.stack([b["protein_label"] for b in batch]), # [B, C]
        "domain_infos": [b["domain_infos"] for b in batch],               # List[List[dict]]
        "seq_len": [b["seq_len"] for b in batch],                         # List[int]
        "uniprot_acc": [b["uniprot_acc"] for b in batch],                 # List[str]
    }
