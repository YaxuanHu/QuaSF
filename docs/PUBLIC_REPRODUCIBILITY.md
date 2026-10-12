# Public reproduction guide

For a small runnable example, start with [the four-protein demo](FOUR_PROTEIN_DEMO.md).
It supplies all four sequences and original features from the final structure figure,
an executable CPU inference command, and verified TSV/JSON output. The remainder of
this guide covers the larger release layout, cached scoring and full evaluation.

## 1. Download and lay out assets

Run from the repository root. `gh` is the optional GitHub CLI; browser downloads work too.

```bash
mkdir -p downloads assets
export ASSETS="$PWD/assets"
gh release download v0.2.0-paper-20261012 --repo YaxuanHu/QuaSF --dir downloads
(cd downloads && sha256sum -c SHA256SUMS.txt)
for name in dataset-manifests smoke-features paper-predictions matched-region-cache; do
  tar -xzf "downloads/$name.tar.gz" -C "$ASSETS"
done
mkdir -p "$ASSETS/weights"
cp downloads/quasf-selected.ckpt downloads/segmentation-source.ckpt downloads/direct-query.pt downloads/region-mlp.pt "$ASSETS/weights/"
```

The first three model files are all needed by the unchanged original restoration function. Do not substitute a checkpoint from a different epoch. `configs/paper_selected.json` records immutable hashes. All release archives contain relative paths under ASSETS.

## 2. CPU inference smoke check

The included validation sample is A0A044RE18. This example strictly loads the paper weights and reports tensor shapes; it does not estimate dataset accuracy. Limit threads/CPU affinity to the allocation of your machine.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python - <<'PY'
import os
from pathlib import Path
import torch
from quasf.paper_train import restore
from quasf.data import EndToEndDomainFunctionDataset
root = Path(os.environ['ASSETS'])
torch.set_num_threads(1)
module = restore(root, root/'weights/direct-query.pt', root/'weights/segmentation-source.ckpt', 'direct')
checkpoint = torch.load(root/'weights/quasf-selected.ckpt', map_location='cpu', weights_only=False)
module.load_state_dict(checkpoint['state_dict'], strict=True)
module.eval()
data = EndToEndDomainFunctionDataset(root/'data/dataset_v2/end_to_end_domain_function_v1/val.jsonl', max_len=1495, data_root=str(root))
item = data[0]
with torch.inference_mode():
    output = module.model(item['residue_feat'][None], item['attn_mask'][None], training=False)
print(item['uniprot_acc'], {k: list(output[k].shape) for k in ['region_masks','region_logits','region_valid']})
PY
```

`region_logits.sigmoid()` gives the 256 scores. Use threshold 0.27 and `region_valid` to exclude padding. Dataset domain segments are 1-based inclusive; emitted prediction segments are 0-based half-open.

## 3. Re-score saved results without GPUs

These commands operate on original saved predictions and caches, not new inference. They allow checking paper arithmetic without downloading every residue embedding.

The historical Region MLP script expects its weight at a fixed asset-relative path:

```bash
mkdir -p "$ASSETS/output/region_mlp_baseline"
cp "$ASSETS/weights/region-mlp.pt" "$ASSETS/output/region_mlp_baseline/region_mlp.pt"
python analysis/region_mlp_trial_20261004/trial.py --root "$ASSETS" --output "$ASSETS/output/region_mlp_trial_20261004" --workers 1
cat "$ASSETS"/output/query_no_mlp_joint_20260928_b32_batched/unified_evaluation/direct/test/predictions_rank*.jsonl > "$ASSETS/test_predictions.jsonl"
python analysis/joint_function_query_20260924/rescore_restored_localization_table.py --predictions "$ASSETS/test_predictions.jsonl" --source "$ASSETS/data/dataset_v2/end_to_end_domain_function_v1/test.jsonl" --output results/table1
python analysis/joint_function_query_20260924/rescore_restored_joint_table.py --predictions "$ASSETS/test_predictions.jsonl" --source "$ASSETS/data/dataset_v2/end_to_end_domain_function_v1/test.jsonl" --cohort analysis/paper_results/heterogeneous_cohort.csv --output results/table2
```

Table 1 above re-scores QuaSF only. The original baseline CSVs are included in `analysis/baseline_predictions/`. Supply them with repeated `--baseline NAME=CSV` to compare all methods. The repository retains baseline result evidence; this release does not regenerate every external baseline.

## 4. Full test inference

The original evaluator requires four explicitly allocated GPUs, original residue features for all test proteins, and copied validation selection records. It uses BF16 and batch 8 per GPU. CPU FP32 smoke output is not a replacement for this numerical recipe.

Prepare a joint directory with `direct/manifest.json`; set its `mask_checkpoint` and `query_checkpoint` to absolute paths of the released weights. Copy `docs/paper_selected_val.json` into a NEW evaluation directory as `direct/selected_val.json` and set only its `joint_checkpoint` field to the absolute path of `quasf-selected.ckpt`. Preserve the checkpoint hashes, threshold, split and partial flag. Do not copy the archived `test/summary.json` into a fresh inference directory, because it represents an already completed historical run.

```bash
python -m torch.distributed.run --nproc_per_node=4 evaluate.py --root "$ASSETS" --joint-output "$JOINT" --output "$NEW_EVALUATION" --split test --batch-size 8
```

`CUDA_VISIBLE_DEVICES` must identify exactly the allocated GPUs. For remote CPU-intensive work, put all jobs in a shared `taskset` CPU set strictly smaller than 70% of the allowed CPUs and set numerical thread limits. On the audited 96-CPU allocation we used CPUs 0-66 (67/96), threads 1. Do not copy that affinity blindly to another machine.

Only the selected checkpoint is released. Repeating validation selection requires all six original epoch checkpoints; the evaluator deliberately rejects a full selection run with only one checkpoint. Training commands and historical protocols are in `REPRODUCIBILITY.md`. Rebuilding the entire raw annotation pipeline requires original TED/Pfam2GO/mapping inputs not supplied here.

## 5. ESM feature preparation

`code/data_preparation/precompute_test_esm_v2.py` is the recorded feature extractor, with 1022-residue windows and 128 overlap. Supply a local ESM2-650M Hugging Face-format model via `--model`, the release test manifest via `--manifest`, and `--project-root "$ASSETS"`. It supports `--shard-index` and `--num-shards`; run one disjoint shard per allocated GPU and use all allocated GPUs. The default model path is historical and must be overridden. Re-encoded features may differ from the original cached environment; use the supplied smoke features for exact artifact-level checks.

## 6. Historical metadata

`docs/RELEASE_PROVENANCE.json` and older release documents retain original experiment paths and hashes. They describe provenance, not required installation paths. `MANIFEST_SHA256.json` describes the current source package. Do not run the historical `scripts/build_release.py` to publish this expanded release: its September allowlist omits newer figure and comparator assets.
