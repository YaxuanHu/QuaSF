# 四条蛋白质示例 / Four-protein demo

从论文最后一张结构图的 **Q7SDR1、A1L0T0、A9KIG5、Q7SIG6** 出发，加载发布权重，实际推理并输出区域边界和功能概率。无需 GPU、训练集、完整测试集或参考标签。

提供的 [sequences.fasta](../examples/sequences.fasta) 是这四条完整序列。QuaSF 的输入是 ESM2 残基特征，因此小型示例包同时提供这四条序列对应的原始预计算特征。**本命令不是任意 FASTA 的在线特征提取器**；更换序列需要重新生成匹配的 ESM 特征。

完整安装指南见 [INSTALL.md](../INSTALL.md)，全部直接软件依赖及可选外部工具见 [DEPENDENCIES.md](../DEPENDENCIES.md)。支持使用 pip 从本仓库源码安装；当前不宣称已发布到 PyPI 或 Bioconda。

## 1. 安装

以下是 Linux/macOS 的 Bash 命令，建议 Python 3.10。已有可用环境的用户可跳过安装，但仍须完成第 2 步准备输入和权重。Windows 可用 WSL 运行下列命令。

```bash
git clone https://github.com/YaxuanHu/QuaSF.git
cd QuaSF
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Linux CPU wheel; macOS 可改为 python -m pip install torch==2.5.0
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

## 2. 下载输入和权重

在仓库根目录运行。示例包约 12 MB，三个必需权重合计约 2.95 GB；建议预留至少 8 GB 内存和 6 GB 磁盘空间。三个权重用于兼容论文原有严格加载流程，不需要下载其他基线权重。

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

解压后的 `demo-assets/paper-examples/` 包含 `sequences.fasta`、不带参考标签的 `records.jsonl`、`manifest.json` 和 `features/` 下的四个特征文件。运行程序会校验输入文件和三个权重的 SHA256，避免文件损坏或使用错版本。

## 3. 一条命令运行

在仓库根目录运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python demo.py --examples demo-assets/paper-examples --weights weights --output demo-output
```

默认且仅使用 CPU FP32，逐条推理，PyTorch 计算线程为 1；无需指定 GPU。不要使用需要四张 GPU 的 `quasf-evaluate` 来运行这个示例。

成功时打印：

```text
Q7SDR1: 390 residues, 2 regions
A1L0T0: 632 residues, 3 regions
A9KIG5: 577 residues, 4 regions
Q7SIG6: 958 residues, 5 regions
SUCCESS: fresh predictions for all four proteins saved in demo-output
```

## 4. 查看输出

```bash
cat demo-output/regions.tsv
```

- `regions.tsv`：每个预测区域的边界、功能类别索引、已恢复的 cluster ID 和概率。一个区域可对应多行功能预测。
- `predictions.json`：全部 14 个区域的 256 维功能概率、阈值、输入序列哈希、权重哈希、精度和运行时间。
- [examples/expected/regions.tsv](../examples/expected/regions.tsv) 和 [predictions.json](../examples/expected/predictions.json)：2026-10-12 实际 CPU 推理输出，供比较；运行程序不读取这些预存结果。

边界采用 **1-based inclusive**（首尾残基都包含）。例如 `90-387` 表示第 90 至第 387 位；多个不连续片段以分号分隔。区域编号遵循模型解码顺序，不一定按序列位置排序。功能采用 sigmoid 多标签概率，验证集确定的阈值为 **0.27**。若某区域无概率达到阈值，TSV 会保留最高概率类别并标记 `above_threshold=False`，它不是阈值以上的预测。

功能标签是训练时的 cluster 类别，不是完整 GO 注释。已恢复的词表只覆盖 256 类中的 188 类；无法恢复的 ID 标为 `unknown`，不猜测类别名称。可对照 [observed_label_vocabulary.json](../configs/observed_label_vocabulary.json) 和论文图示中的功能说明。

## 实测范围

四条输入全部从示例压缩包解压，以发布的三个权重严格加载并完成真实前向推理，产生 2/3/4/5 个区域及有限功能概率。验证使用现有 Python 3.10.12、PyTorch 2.5.0+cu124 环境中的 **CPU**；不是重新创建环境安装全部依赖的验证，也不是 9,017 条完整测试集重跑。全新环境安装流程列于上方，未将其声称为已实测。

CPU FP32 与论文 GPU BF16 的浮点概率可能有细小差异；本示例用于展示模型真实输入输出，不要求不同硬件逐位一致。历史图源若使用 0-based half-open 坐标，转换到本例格式时起点加 1、终点不变。论文图中只选取部分功能展示，TSV 保留所有达到阈值的类别。

遇到 `Missing ...ckpt` 请检查第 2 步下载目录；遇到 checksum mismatch 请重新下载对应文件；`No module named quasf` 时请确认处于仓库根目录且已执行 `python -m pip install -e .`。PyTorch 的 nested tensor prototype 提示不影响成功输出。
