# Baseline Results on `data/baseline_csvs/test.csv`

Full unified test set: **27,904 proteins** (one protein had no ground-truth domains and is excluded from the metrics, leaving 27,903 evaluated proteins).

Metrics:
- **IoU** – domain-length weighted Intersection over Union between predicted and ground-truth domains (Hungarian matching).
- **MCC** – domain-boundary Matthews Correlation Coefficient with ±20 residue tolerance.
- Both metrics are reported as **domain-length weighted** averages and as **simple per-protein averages**.

## Result summary

| method            | status              | #proteins | wIoU   | wMCC   | avg IoU | avg MCC | note |
|-------------------|---------------------|-----------|--------|--------|---------|---------|------|
| unidoc_auto       | retrained           | 27,550    | 0.7636 | 0.4121 | 0.7779  | 0.4092  | trained on our train set |
| sword             | tuned               | 27,903    | 0.5538 | 0.3676 | 0.5288  | 0.3697  | hyperparams tuned on our train set |
| unidoc            | original_alg        | 27,897    | 0.4291 | 0.3334 | 0.4484  | 0.3038  | non-learning baseline |
| chainsaw          | abandoned           | 0         | –      | –      | –       | –       | too slow for full retraining (~40+ days) |
| eguchi_cnn        | abandoned           | 0         | –      | –      | –       | –       | massive cache I/O bottleneck; first epoch did not start |
| merizo            | external            | 0         | –      | –      | –       | –       | user ran elsewhere |
| **chainsaw_official** | **official_pretrained** | **27,903** | **0.8766** | **0.7981** | **0.8799** | **0.7976** | **official pretrained weights, zero-shot** |
| **merizo_official**   | **official_pretrained** | **27,903** | **0.8089** | **0.5496** | **0.8177** | **0.5318** | **official pretrained weights, zero-shot, no-iterate, 24 GB GPU** |

## Notes on official baselines

### Chainsaw official
- Model: `code/baselines/chainsaw-main/saved_models/model_v3/weights.pt`
- Ran on GPU 0 over the full test set.
- Zero failures; all 27,903 proteins with ground-truth domains were evaluated.
- Runtime: ~3 h 6 min (~2.5 proteins/sec).

### Merizo official
- Model: `code/baselines/merizo/weights/weights_part_*.pt`
- Ran in **non-iterative** mode (`--iterate=False`, `--length_conditional_iterate=False`) to fit in 24 GB GPU memory.
- Split into three index chunks and processed in parallel on GPUs 1, 2 and 3.
- Four proteins initially failed because of an empty-array truth-value ambiguity in the official PDB parser (`model/utils/pdb_parser.py`). The parser was patched to check `.size > 0` and the proteins were retried successfully, bringing the evaluated count from 27,899 to 27,903.
- Runtime per chunk: ~2 h 15–2 h 30 min.

## Files

- `all_baseline_results.csv` – combined summary table.
- `chainsaw_official_results.csv` – Chainsaw official summary.
- `chainsaw_official_failed.txt` – empty (no failures).
- `merizo_official_gpu{1,2,3}_results.csv` – per-chunk summaries.
- `merizo_official_gpu{1,2,3}_results_per_protein.csv` – per-protein metrics for exact recombination.
- `merizo_official_failed.txt` – consolidated failure log.
- `merge_official_results.py` – script that merges the Merizo per-protein chunk files and updates `all_baseline_results.csv`.
