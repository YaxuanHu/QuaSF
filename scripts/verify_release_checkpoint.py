"""Compare the release with original remote code using the selected paper weights."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import pytorch_lightning as pl
import torch

from quasf.data import EndToEndDomainFunctionDataset
from quasf.evaluate import sha256
from quasf.paper_train import restore


def verify(args: argparse.Namespace) -> dict:
    """Strictly restore the real weights and compare every forward tensor exactly."""
    torch.set_num_threads(1)
    selected = json.loads((Path(__file__).resolve().parents[1] / "configs/paper_selected.json").read_text())
    identity = sha256(args.joint_checkpoint)
    if identity != selected["checkpoint_sha256"]:
        raise ValueError("joint checkpoint differs from the selected paper checkpoint")
    spec = importlib.util.spec_from_file_location(
        "archived_quasf", args.reference_release / "quasf/__init__.py",
        submodule_search_locations=[str(args.reference_release / "quasf")])
    package = importlib.util.module_from_spec(spec)
    sys.modules["archived_quasf"] = package
    spec.loader.exec_module(package)
    original = importlib.import_module("archived_quasf.paper_train")
    left = original.restore(args.root, args.query_checkpoint, args.mask_checkpoint, "direct")
    right = restore(args.root, args.query_checkpoint, args.mask_checkpoint, "direct")
    checkpoint = torch.load(args.joint_checkpoint, map_location="cpu", weights_only=False)
    for module in (left, right):
        module.load_state_dict(checkpoint["state_dict"], strict=True)
        module.eval()
    del checkpoint
    if list(left.state_dict()) != list(right.state_dict()):
        raise AssertionError("checkpoint parameter namespaces differ")
    dataset = EndToEndDomainFunctionDataset(
        args.root / "data/dataset_v2/end_to_end_domain_function_v1/val.jsonl",
        max_len=1495, data_root=str(args.root))
    item = dataset[args.record_index]
    features, mask = item["residue_feat"][None], item["attn_mask"][None]
    with torch.inference_mode():
        before = left.model(features, mask, training=False)
        after = right.model(features, mask, training=False)
    if before.keys() != after.keys():
        raise AssertionError("forward outputs differ")
    checked = []
    for key in before:
        if not torch.is_tensor(before[key]):
            raise AssertionError(f"unexpected output type: {key}")
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, msg=key)
        checked.append(key)
    result = {
        "environment": "server221 CPU verification of pretrained direct release", "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__, "pytorch_lightning": pl.__version__,
        "source_root": str(args.root),
        "release_root": str(Path(__file__).resolve().parents[1]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "numerical_threads": 1,
        "gpu_used": False, "selected_checkpoint_sha256": identity,
        "mask_checkpoint_sha256": sha256(args.mask_checkpoint),
        "query_checkpoint_sha256": sha256(args.query_checkpoint),
        "strict_checkpoint_load": "passed for both original and release",
        "state_dict_keys": len(right.state_dict()),
        "validation_record_index": args.record_index, "accession": item["uniprot_acc"],
        "visible_residues": int(item["seq_len"]), "input_shape": list(features.shape),
        "forward_tensors_compared": checked, "rtol": 0, "atol": 0,
        "forward_parity": "all tensors exactly equal",
        "scope": "one real validation protein; no retraining or full-split reevaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    return result


def main() -> None:
    """Run as python -m scripts.verify_release_checkpoint from the release root."""
    parser = argparse.ArgumentParser()
    for name in ("root", "mask_checkpoint", "query_checkpoint", "joint_checkpoint", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--reference-release", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=0)
    print(json.dumps(verify(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
