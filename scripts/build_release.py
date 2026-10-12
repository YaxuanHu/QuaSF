"""Build an independent publication source archive from an explicit file list."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build_release(output: Path) -> Path:
    """Include publication source, tests, protocol and source identities only."""
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in (
        "README.md", "ARCHITECTURE.md", "pyproject.toml", "requirements.txt",
        "requirements-features.txt", "train.py", "evaluate.py", "pretrain.py", "segmentation_pretrain.py",
        "docs/REPRODUCIBILITY.md", "docs/RELEASE.md", "docs/RELEASE_PROVENANCE.json",
        "docs/READ_CODE.md", "docs/CLEANUP.json",
        "docs/paper_selected_val.json", "docs/paper_test_summary.json",
        "configs/paper_selected.json", "scripts/check.py", "scripts/build_release.py",
        "scripts/verify_release_checkpoint.py",
    )]
    files.extend(sorted((root / "quasf").glob("*.py")))
    files.extend(sorted((root / "tests").glob("*.py")))
    for folder in ("code", "analysis"):
        files.extend(sorted(path for path in (root / folder).rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".md", ".sh", ".csv"} and "__pycache__" not in path.parts))
    verification = root / "docs/RELEASE_VERIFICATION.json"
    if verification.exists() and json.loads(verification.read_text())["selected_checkpoint_sha256"] == json.loads((root / "configs/paper_selected.json").read_text())["checkpoint_sha256"]:
        files.append(verification)
    manifest = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files}
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "QuaSF/" + path.relative_to(root).as_posix())
        archive.writestr("QuaSF/RELEASE_FILES.json", json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output)
    return output


def main() -> None:
    """Expose the source archive builder."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "dist/QuaSF-source.zip")
    args = parser.parse_args()
    print(build_release(args.output))


if __name__ == "__main__":
    main()
