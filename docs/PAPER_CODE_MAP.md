# Paper-to-code map (online manuscript checked 2026-10-12)

Paper: QuaSF: Protein Domain Segmentation and Function Prediction with Disentangling Queries.
Online results and methods were read on 2026-10-12. Manuscript source is kept in the private release audit, not redistributed in this repository.

| Paper component | Code / evidence |
|---|---|
| Segmentation backbone | `quasf/model.py`, `quasf/segmentation_train.py`, `segmentation_pretrain.py` |
| Per-final-region function queries | `quasf/model.py`: `DirectFunctionQuery`, `BatchedJointFunctionQuery` |
| Reference-region pretraining | `quasf/pretrain.py`, `quasf/region_cache.py`, `pretrain.py` |
| Adaptation to predicted regions | `quasf/paper_train.py`, `quasf/train.py`, `train.py` |
| Validation selection and strict evaluation | `quasf/paper_evaluate.py`, `quasf/metrics.py`, `evaluate.py` |
| Table 1, continuous localization | `analysis/joint_function_query_20260924/rescore_restored_localization_table.py` |
| Table 2, heterogeneous regional recovery | `analysis/joint_function_query_20260924/rescore_restored_joint_table.py`; `analysis/reviewer_checks_20260914/`; `analysis/paper_results/` |
| Shared-query comparator | `code/shared_query_ablation/` |
| Region MLP training | `analysis/region_mlp_trial_20261004/region_mlp.py` |
| Matched-region F1 and MLP preferences | `analysis/region_mlp_trial_20261004/trial.py`, `summary.json` |
| P43561 MLP error case | `analysis/section34_revision2/`: figure builder, saved case records, candidate search and scores |
| Three prediction mechanisms | `analysis/query_vertical_ppt/`: editable PPTX, PDF and original PowerShell builder (requires PowerPoint) |
| Four structural examples | `analysis/joint_function_query_20260924/build_structure_cards_v5.py` and accompanying manifests |
| Dataset construction / ESM features | `code/data_preparation/`; historical input tables are external |
| External segmentation baselines / M2 | `code/baselines/` |

The 2026-09-30 source package omitted Region MLP because it was absent from that paper version. It is restored as a separate comparator for the current manuscript; it is not part of the QuaSF model.

## Metric and cohort distinctions

- Raw held-out split: 9,017 proteins. Continuous-localization cohort: 9,015 (excludes P06868 and Q58969).
- Matched annotated regions: 11,034. QuaSF F1 = 0.8836143674 at validation threshold 0.27; Region MLP F1 = 0.8404937372 at validation threshold 0.45. Unmatched regions are excluded from this metric.
- Regional preferences: QuaSF 2,430/2,444, MLP 2,401/2,444 on the same 1,277 proteins; shared query uses its own matched set, 20/2,454 on 1,276 proteins.
- Strict end-to-end F1 = 0.486798215 includes localization failures and must not be substituted for matched-region F1.
- Heterogeneous recovery uses 1,376 proteins, 4,303 reference components, and 5,606 region-function pairs. Equal output budget is 9,149 calls.
- Selection occurred in the first adaptation epoch while segmentation parameters were frozen.

## Scope

Scientific model files are unchanged from the verified 2026-09-30 source. Additional scripts are original archived scripts, with historical paths retained for traceability. `docs/PUBLIC_REPRODUCIBILITY.md` explains portable asset paths and limitations. Original `docs/RELEASE*`, `docs/CLEANUP.json`, and `RELEASE_FILES.json` describe the older source snapshot, not this expanded release.
