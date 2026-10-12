# Current paper reproduction protocol (2026-09-30)

## Selection and external assets
ASSETS on server221 is /data/huyaxuan/code/region_function_multimodal.
The recorded server221 runtime is /data/huyaxuan/miniconda3/envs/dom/bin/python.
Its existing Conda build needs PYTHONPATH=/data/huyaxuan/code/compat_221 for the
platform-version compatibility shim used by the original experiment. Do not add
the experiment vendor directory to PYTHONPATH when running this standalone release.
Data: data/dataset_v2/end_to_end_domain_function_v1/{train,val,test}.jsonl;
72,107 / 9,014 / 9,017 proteins, 256 classes, max length1495, ESM dimension1280.
The original server220 ESM feature version is retained.

- Mask source: output/mask_condition_ablation/mask_resume_20260922_101745/checkpoints/best_tri-epoch=12-val_tri_score=0.779311.ckpt
- Pretrained direct query: output/query_no_mlp_ablation_20260928/direct/best.pt
- Continuation root: output/query_no_mlp_joint_20260928_b32_batched
- Selected checkpoint: direct/checkpoints/epoch=0-step=564.ckpt within that root.
- SHA256: d355049c93c8a94fab9fd5c78772edf8053e7593b3f7b2494a75607bd4c381fe.
- Function threshold0.27, mask threshold0.3. Selection is during functional warmup.
- Recorded complete evaluation: docs/paper_selected_val.json and paper_test_summary.json.

Reference-region features/labels/group metadata are shards in
output/region_mlp_baseline/cache/{train,val}. This is a historical cache name;
pretraining does not require the MLP checkpoint. Cache helpers remain in
quasf/region_cache.py. Data, caches and checkpoints are not bundled.

## Training commands
The recorded recipe uses four allocated GPUs, DDP, seed42: 12 query-pretraining
epochs at global batch64, then six continuation epochs at global batch128 (32 per
GPU), first epoch warmup. Use new output directories. CUDA_VISIBLE_DEVICES must
already identify the four explicitly allocated GPUs.

Before CPU-intensive remote work use taskset with fewer than70% of allowed CPUs,
shared across our concurrent jobs, and numerical threads1. For allowed CPUs0-95,
0-65 satisfies this policy; adapt the set if the allocation differs.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
taskset -c 0-65 python -m torch.distributed.run --nproc_per_node=4 pretrain.py --root "$ASSETS" --output "$NEW_PRETRAIN" --epochs 12 --global-batch 64
taskset -c 0-65 python train.py --root "$ASSETS" --output "$NEW_JOINT" --query-checkpoint "$NEW_PRETRAIN/direct/best.pt" --batch-size 32
```

Continuation reads the recorded mask source under ASSETS. All six checkpoints
are retained. Internal Lightning validation does not replace final unified
validation selection. Do not attribute the selected warmup result to later
joint segmentation updates.

## Complete validation before test
The evaluator preserves the four-rank BF16 recipe with batch8 per GPU. Source
checkpoint paths come from the continuation manifest.json; if migrating assets,
update paths in a copy of that manifest, without changing the asset identities.

```bash
taskset -c 0-65 python -m torch.distributed.run --nproc_per_node=4 evaluate.py --root "$ASSETS" --joint-output "$NEW_JOINT" --output "$NEW_EVALUATION" --split val --batch-size 8
taskset -c 0-65 python -m torch.distributed.run --nproc_per_node=4 evaluate.py --root "$ASSETS" --joint-output "$NEW_JOINT" --output "$NEW_EVALUATION" --split test --batch-size 8
```

For paper weights JOINT is $ASSETS/output/query_no_mlp_joint_20260928_b32_batched.
Use a new evaluation output. Test scoring is bound to complete validation and
checkpoint hash. Partial --limit runs cannot select a model. No GPU is required
for source inspection or CPU release checks.

Strict evaluation preserves domain identity, requires IoU>=0.5 and an observed
regional label. Test F1 uses validation threshold0.27; test-scanned Fmax is a
different statistic. Regional preferences are within-protein comparisons, not
whole-test protein accuracy. Unknown biological labels remain unknown. The9015
continuous-localization and1376 annotation-recovery protocols are separate from
the9017 identity-preserving evaluation.

Segmentation source training is retained in quasf/segmentation_train.py, exposed
by segmentation_pretrain.py. Use the recorded source hyperparameters and original
resume checkpoint when reproducing its continuation; the direct-model stage
restores the immutable source checkpoint identified above.
