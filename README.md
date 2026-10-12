# QuaSF

**Protein Domain Segmentation and Function Prediction with Disentangling Queries**

QuaSF predicts protein regions and assigns a separate function query to each final region, using frozen ESM2 residue features. This repository contains the paper model, training/evaluation code, paper baselines, and figure/metric provenance. The October 12 release includes the Region MLP comparison and P43561 case in the current manuscript.

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
