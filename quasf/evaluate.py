"""Validation-fixed scoring helpers used by the paper evaluator."""
from pathlib import Path
import hashlib
from .metrics import Fmax

def sha256(path: Path) -> str:
    """Bind results to immutable model and split inputs."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def fixed_macro_f1(state: Fmax, threshold: float) -> float:
    """Return macro F1 at an already selected validation threshold."""
    index = int(round(threshold * 100))
    if abs(float(state.thresholds[index]) - threshold) > 1e-8:
        raise ValueError("threshold must be on the 0.01 validation grid")
    precision = state.p[index] / state.n
    recall = state.r[index] / state.n
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
