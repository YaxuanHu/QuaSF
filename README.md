# QuaSF

**Protein Domain Segmentation and Function Prediction with Disentangling Queries**

QuaSF predicts protein regions and assigns a separate function query to each final region, using frozen ESM2 residue features. This repository contains the paper model, training/evaluation code, paper baselines, and figure/metric provenance. The October 12 release includes the Region MLP comparison and P43561 case in the current manuscript.

## 四条蛋白质快速开始 / Four-protein CPU demo

测试论文最后一张结构图中的 **Q7SDR1、A1L0T0、A9KIG5、Q7SIG6**。提供四条完整序列及对应 ESM 特征；下载三个发布权重后即可在 CPU 上实际推理，无需 GPU、训练数据或完整测试集。

以下命令适用于 Linux Bash（Windows 可用 WSL），建议 Python 3.10。**所有命令在仓库根目录 `QuaSF/` 执行**。示例特征包约 12 MB，三个权重合计约 2.95 GB。

### 1. 获取代码并安装

```bash
git clone https://github.com/YaxuanHu/QuaSF.git
cd QuaSF
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

### 2. 下载权重与四条示例输入

```bash
mkdir -p weights demo-assets
BASE=https://github.com/YaxuanHu/QuaSF/releases/download/v0.2.0-paper-20261012
for FILE in quasf-selected.ckpt segmentation-source.ckpt direct-query.pt; do
  curl -fL --retry 3 "$BASE/$FILE" -o "weights/$FILE" || exit 1
done
curl -fL --retry 3 \
  https://github.com/YaxuanHu/QuaSF/releases/download/v0.2.1-examples/paper-examples.tar.gz \
  -o demo-assets/paper-examples.tar.gz
python -c "import hashlib; from pathlib import Path; p=Path('demo-assets/paper-examples.tar.gz'); assert hashlib.sha256(p.read_bytes()).hexdigest() == 'd62c08b29c1de7e818af44a8cb0baebf0d93815dec61d117178f855e21cd3071', 'Example download checksum mismatch'"
tar -xzf demo-assets/paper-examples.tar.gz -C demo-assets
```

目录准备完成后应有：

```text
QuaSF/
├── demo.py
├── weights/
│   ├── quasf-selected.ckpt
│   ├── segmentation-source.ckpt
│   └── direct-query.pt
├── demo-assets/paper-examples/
│   ├── sequences.fasta
│   ├── records.jsonl
│   ├── manifest.json
│   └── features/                 # 四个 .pt 特征文件
└── examples/expected/            # 已验证输出，仅供对照
```

### 3. 运行并查看结果

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python demo.py --examples demo-assets/paper-examples --weights weights --output demo-output
cat demo-output/regions.tsv
```

实测成功输出：

```text
Q7SDR1: 390 residues, 2 regions
A1L0T0: 632 residues, 3 regions
A9KIG5: 577 residues, 4 regions
Q7SIG6: 958 residues, 5 regions
SUCCESS: fresh predictions for all four proteins saved in demo-output
```

`demo-output/regions.tsv` 给出区域边界、功能类别和概率；`demo-output/predictions.json` 保存全部 14 个区域的 256 维功能概率和运行信息。边界为 **1-based inclusive**，功能阈值为 **0.27**，一个区域可以有多个功能预测。类别为 cluster ID，已恢复词表并非完整 GO 注释。CPU FP32 的细小浮点差异是允许的。

四条序列可直接查看 [examples/sequences.fasta](examples/sequences.fasta)，实际结果见 [examples/expected/regions.tsv](examples/expected/regions.tsv)。程序每次执行真实前向推理，不读取 `expected/`。输入依赖随包提供的 ESM 特征，更换 FASTA 需要重新生成匹配特征。

[examples 子目录说明](examples/README.md)同样包含安装、下载、运行、输出解释，并给出了**从子目录直接运行**的相对路径命令。[完整使用说明](docs/FOUR_PROTEIN_DEMO.md)还包含校验、常见错误、macOS 安装差异和实测范围。

## Install and check

Python 3.10 is the recorded runtime. Install the appropriate PyTorch build for your machine, then:

```bash
python -m pip install -e .
python scripts/check.py
```

The eight checks cover parameter initialization, output/gradient parity, region identity, empty regions, strict metrics, and source preservation. Figure builders additionally need matplotlib and PyMuPDF; PPTX builders require Windows PowerPoint.

## Weights and data

Download files from [v0.2.0-paper-20261012](https://github.com/YaxuanHu/QuaSF/releases/tag/v0.2.0-paper-20261012). Check `SHA256SUMS.txt` before use.

| Download | Contents |
|---|---|
| `quasf-selected.ckpt` | Selected QuaSF checkpoint, epoch 0 / step 564 |
| `segmentation-source.ckpt` | Segmentation checkpoint required by the original restoration path |
| `direct-query.pt` | Pretrained regional function query required by restoration |
| `shared-query.ckpt` | Validation-selected shared-query comparator |
| `shared-query-predictions.tar.gz` | Saved shared-query validation/test predictions |
| `region-mlp.pt` | Separate Region MLP comparator |
| `dataset-manifests.tar.gz` | Train/validation/test sequences, regions, labels and fixed common-test cohort |
| `smoke-features.tar.gz` | Original ESM features for A0A044RE18, P43561 and Q7SDR1 |
| `paper-predictions.tar.gz` | Saved selected validation/test predictions and evaluation records |
| `matched-region-cache.tar.gz` | Saved matched-region features, labels, scores and comparison script |

Data annotations retain upstream provenance. Their license is not replaced by the source-code MIT license. Full original validation/test ESM features are preserved in the server release, but are not included in these GitHub assets. The feature generation script is included; re-encoding in another environment is not guaranteed bitwise identical. Training residue features, original full GO-to-cluster construction inputs, full baseline weights and external structure models are not bundled. The four saved structure renders needed to rebuild the paper panel are included.

Follow [PUBLIC_REPRODUCIBILITY.md](docs/PUBLIC_REPRODUCIBILITY.md) for asset layout, a CPU inference example, cached numerical reproduction and full evaluation requirements.

## Paper results

| Evaluation | QuaSF |
|---|---:|
| Continuous localization weighted IoU (9,015 proteins) | 0.8362 |
| Matched annotated-region function F1 | 0.8836 |
| Within-protein preference accuracy | 2,430/2,444 = 99.43% |
| Heterogeneous recovery, correct calls at budget 9,149 | 5,326 |

The raw test manifest has **9,017** proteins; the localization cohort excludes P06868 and Q58969. Matched-region F1 excludes unmatched regions and differs from strict end-to-end F1 (0.4868). Thresholds are validation-selected. The selected adaptation epoch keeps segmentation frozen. Region MLP is a separate baseline with different training, not an architectural ablation with otherwise matched training.

## Code

- `quasf/`: model, losses, data adapters, training and evaluation.
- `train.py`, `pretrain.py`, `segmentation_pretrain.py`, `evaluate.py`: entry points.
- `code/`: paper baselines, shared-query comparator, historical dataset construction.
- `analysis/`: table scoring, Region MLP comparison, P43561 and structural figure sources.
- `configs/`, `docs/`, `tests/`: settings, provenance, protocols and checks.

See [paper-to-code map](docs/PAPER_CODE_MAP.md). Historical experiment scripts retain their original paths; consult their arguments before running. The main package has no dependency on private experimental modules.

## Verification and limitations

The release passed all eight source/numerical checks and restored the selected checkpoint strictly (207 state-dict entries). All 16 forward output tensors match the original release exactly on one real validation protein (CPU FP32). This is a smoke/parity check, not a fresh full-split GPU evaluation. Cached-result rescoring is reported separately in `docs/verification/`.

`configs/observed_label_vocabulary.json` recovers observed cluster labels and GO names from the saved annotations. It covers 188 observed classes; it does not reconstruct the unavailable full 256-class GO mapping. Model indices remain unchanged.

## License

Original QuaSF code: MIT. Third-party baseline code retains its original licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset and external-model terms are separate.
