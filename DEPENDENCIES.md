# Software dependency inventory

Audited against the distributed Python imports and external-tool invocations on
2026-10-13. The required runtime is separated from optional research workflows.
For installation commands and supported/verified environments see [INSTALL.md](INSTALL.md).

## Core QuaSF and four-protein CPU example

All direct runtime dependencies are declared in [requirements.txt](requirements.txt)
and consumed by [pyproject.toml](pyproject.toml) during installation.

| Distribution | Version | Purpose |
|---|---|---|
| torch | 2.5.0 | Model, checkpoint loading, tensors and inference/training |
| pytorch-lightning | 2.6.5 | Training and original checkpoint restoration |
| numpy | 2.2.6 | Numerical arrays, decoding and metrics |
| scipy | 1.15.3 | Assignment, optimization and scientific metrics |
| scikit-learn | 1.7.2 | Evaluation metrics; import name `sklearn` |

Python 3.10 is the recorded runtime. Build backend: **setuptools >=68**, installed
by pip's isolated build from `pyproject.toml`. Source checks use standard-library
`unittest`; pytest is not required. The core model has no STRIDE, PyMOL, ESM model
or PowerPoint dependency when precomputed residue features are supplied.

## ESM feature generation (optional)

[requirements-features.txt](requirements-features.txt): **transformers >=4.30,<4.50**,
in addition to torch from the core runtime. Used by
`code/data_preparation/precompute_test_esm_v2.py` (`AutoModel`, `AutoTokenizer`).
The old `fair-esm` entry did not match this script and has been corrected.
The version range is an installation constraint, not a recovered historical lock.
A compatible local ESM2 650M model/tokenizer, its license, CUDA BF16-capable hardware,
and a sequence manifest are additional inputs. The script uses local-only loading;
installing transformers does not download those assets.

## Figure generation and auditing (optional)

[requirements-analysis.txt](requirements-analysis.txt), in addition to the core runtime:

| Distribution | Constraint | Purpose |
|---|---|---|
| matplotlib | >=3.7,<4 | Plotting and figure assembly |
| Pillow | >=10,<13 | Read saved structure-render PNGs; import `PIL` |
| PyMuPDF | >=1.24,<2 | PDF reading/auditing; imports `fitz` and `pymupdf` |

The structure-panel builder assembles existing renders; it does not run AlphaFold.
Saved renders and figure manifests are included. Original experimental result
files may be required by other figure scripts. The `.ps1` PPTX builders/exporters
under `analysis/query_vertical_ppt/` require **Windows PowerShell and desktop
Microsoft PowerPoint with COM automation**, not a Python package. These are not
needed to run QuaSF or view the supplied PDFs/PPTX files.

## Baselines and historical analysis (optional)

[requirements-baselines.txt](requirements-baselines.txt) adds these Python dependencies
to the core runtime. Constraints are listed in that file; optional workflows have
not all been freshly installed or rerun as part of this packaging audit.

| Distribution | Import | Used for |
|---|---|---|
| pandas | pandas | Baseline manifests, CSV preparation and evaluation |
| biopython | Bio | Protein structure parsing in EguchiCNN/Chainsaw |
| h5py | h5py | EguchiCNN HDF5 cache conversion and reading |
| einops | einops | Merizo tensor rearrangement |
| natsort | natsort | Merizo natural sorting |
| networkx | networkx | Merizo graph operations |
| pydantic | pydantic | Chainsaw result models |
| click | click | Chainsaw archive extraction CLIs |
| requests | requests | Official-weight download helper |
| tqdm | tqdm | Official baseline progress reporting |
| rotary-embedding-torch | rotary_embedding_torch | Merizo rotary positional embeddings (0.6.5) |
| matplotlib | matplotlib | Merizo plots |

Shared-query and regional MLP model code otherwise use the core packages.
Names such as `model`, `src`, `dataset`, `loss`, `evaluate_metrics` and
`v7_joint_common9015` refer to modules in this repository, not packages to install
from PyPI. Historical scripts can require their documented working directories
and explicit data paths; a dependency installation does not repair those paths.

### External software and assets

- **STRIDE executable**: required by the bundled official Chainsaw structure
  featurizer. Obtain it under its upstream license and set `STRIDE_EXE` to the
  executable path. It is not supplied by pip or required by QuaSF itself.
- **PyMOL executable**: needed only for the optional Chainsaw 3D visualization
  helper; not for QuaSF inference or the supplied figure renders.
- Baseline checkpoints and PDB/mmCIF structures are separate input assets.
  The included SWORD/UniDoc Python wrappers use their bundled Python modules;
  this dependency list is not an installer for external upstream applications.
- Git, curl and tar support the documented checkout/download workflow; browser
  download and extraction can substitute for curl/tar. GitHub CLI is optional.
- Linux `taskset` is used by the provided server-specific execution wrapper to
  enforce CPU allocation. It is not imported by the portable Python demo.
- NVIDIA driver/CUDA-capable GPUs are needed only for GPU workflows; use the
  appropriate PyTorch wheel. No local C/CUDA extension compilation is required
  by QuaSF's Python package when binary dependencies are available.

## Indirect dependencies

pip resolves the dependencies of the distributions above (for example Lightning's
`torchmetrics`/`lightning-utilities`, scikit-learn's `joblib`/`threadpoolctl`,
Transformers' tokenizer/model-download libraries, and platform-specific PyTorch
runtime libraries). They vary with platform and optional groups and are **not**
all pinned by this repository. This is a complete direct dependency inventory,
not a universal transitive lockfile. `python -m pip check` validates the installed
set; `python -m pip freeze > installed-environment.txt` records every resolved
Python distribution and version for a particular environment.
