# Installation / 安装指南

QuaSF supports installation **from this GitHub source checkout using pip**.
This project does not currently advertise a PyPI or Bioconda distribution:
`pip install quasf` and `conda install quasf` are not the documented installation routes.
A Conda environment can still be used to install this checkout with pip.

本项目提供源码安装、明确的依赖清单和可运行示例，可按投稿问卷的
“详细安装指南并列出全部软件依赖”这一替代条件回答 Yes；不要填写成已发布到 PyPI/Bioconda。

## 1. System requirements

- Recommended/recorded Python: **3.10** (paper verification: 3.10.12).
  Use this version for reproduction; package metadata's `>=3.10` is not a promise
  that every future Python version has wheels for the pinned dependencies.
- Recommended OS: Linux x86-64. The four-protein example runs on CPU;
  no GPU, CUDA toolkit, ESM download or structure predictor is required for that example.
- Windows: WSL2 is recommended for the Bash commands below. Native Windows and
  macOS are not end-to-end verified by this release.
- Tools: Git to clone; Python with `venv` and pip; `curl` and `tar` for downloading
  and unpacking example assets (browser download/extraction is also possible).
- Four-protein demo: allow at least 8 GB RAM and 6 GB free disk for assets/environment;
  actual needs depend on the chosen PyTorch build and package cache.
- Training/full evaluation: Linux, NVIDIA GPUs and a compatible NVIDIA driver.
  The recorded four-GPU workflow uses PyTorch **2.5.0+cu124**, BF16 and four
  explicitly allocated GPUs. The demo uses CPU and does not require this setup.

## 2. Create a clean environment

Run these commands in Bash:

```bash
git clone https://github.com/YaxuanHu/QuaSF.git
cd QuaSF
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Alternatively, if Conda is already installed, replace the venv commands with:

```bash
conda create -n quasf python=3.10 pip -y
conda activate quasf
```

All following commands run from the `QuaSF/` repository root in that environment.
Do not install the CPU and CUDA variants together; choose one.

## 3. Install QuaSF and its required dependencies

**CPU demo (Linux/WSL2):**

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

**GPU training/evaluation (Linux, compatible NVIDIA driver):**

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

For macOS, install `torch==2.5.0` from the normal package index instead of the CPU
index, subject to wheel availability for your architecture. The demo still uses CPU.
The build choices follow [PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/).
QuaSF does not require torchvision or torchaudio.

`pip install -e .` reads `requirements.txt` through `pyproject.toml`, installs all
five direct runtime dependencies and resolves their transitive dependencies.
An ordinary non-editable installation is also supported: `python -m pip install .`.
Keep the checkout: demo inputs, analysis scripts and documentation live outside
the installed `quasf` package. Model weights and data are downloaded separately.

## 4. Verify installation and run the example

```bash
python -m pip check
python -c "import torch, quasf; print('torch:', torch.__version__); print('quasf:', quasf.__file__)"
python scripts/check.py
python demo.py --help
```

The source/numerical suite should report **8 tests, OK**. Then follow the
[four-protein example guide](docs/FOUR_PROTEIN_DEMO.md#2-下载输入和权重) to download
three weights and the small feature archive, and run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python demo.py --examples demo-assets/paper-examples --weights weights --output demo-output
```

Expected region counts: Q7SDR1 **2**, A1L0T0 **3**, A9KIG5 **4**, Q7SIG6 **5**.
Results are `demo-output/regions.tsv` and `demo-output/predictions.json`.
See [actual verified output](examples/expected/regions.tsv).

## 5. Optional workflows

Install only the groups needed for your work, after the core installation:

```bash
# Regenerate ESM2 features with the included Hugging Face script
python -m pip install -r requirements-features.txt
# Build/audit the paper's Python figures
python -m pip install -r requirements-analysis.txt
# Run the included baseline Python code (also needs method-specific assets/tools)
python -m pip install -r requirements-baselines.txt
python -m pip check
```

[DEPENDENCIES.md](DEPENDENCIES.md) lists every direct third-party Python dependency
found in the distributed Python sources, its role, external applications and
non-software inputs. Installing optional dependencies alone does not supply
baseline weights, structures or historical data paths.

The feature script uses `transformers`, not `fair-esm`, and loads a **local** ESM2
model/tokenizer directory via `--model` with `local_files_only=True`. Its recorded
path uses CUDA BF16. The provided four-protein demo avoids this optional step.

## 6. Troubleshooting and reproducibility

- `No matching distribution`: use Python 3.10 and a platform with PyTorch 2.5.0 wheels;
  do not silently upgrade the pinned scientific dependencies.
- `No module named quasf`: activate the same environment used to install the checkout;
  use `python -m pip` rather than a potentially unrelated `pip` executable.
- A dependency conflict reported by `pip check`: start a clean environment instead
  of mixing this project with an existing scientific installation.
- Missing checkpoints/features or checksum mismatch: complete the example download
  steps and re-download the affected file. Assets are not included in the Python package.
- CUDA not available: the CPU example still works; GPU workflows require the CUDA
  build and driver support. Do not use unallocated GPUs.
- `fitz` missing: install `requirements-analysis.txt`; the distribution name is
  **PyMuPDF**, not the unrelated `fitz` package.

To record all resolved packages, including indirect dependencies, after installing:

```bash
python -m pip freeze > installed-environment.txt
python -m pip check
```

The core versions are pinned from the recorded runtime. The eight source checks
and four-protein inference were verified in the existing Python 3.10 server
runtime. Optional version ranges document installation constraints, not a claim
that every optional baseline/figure pipeline was freshly installed and rerun.
