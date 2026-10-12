"""Shared I/O and provenance helpers; no model imports."""
import hashlib
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(directory):
    root = Path(directory)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def runs(mask):
    """Lossless intervals, zero-based half-open."""
    import numpy as np
    changes = np.flatnonzero(np.diff(np.pad(mask.astype(np.int8), (1, 1))))
    return changes.reshape(-1, 2).astype(int).tolist()
