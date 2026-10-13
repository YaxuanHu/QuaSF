# Paper experiments

Installation: [../INSTALL.md](../INSTALL.md). Dependency inventory: [DEPENDENCIES.md](../DEPENDENCIES.md). From the repository root, install the core package, then `python -m pip install -r requirements-analysis.txt`. Method-specific external tools/assets are listed in the inventory.

Table 1: joint_function_query_20260924/rescore_restored_localization_table.py
accepts --predictions (merged direct JSONL), --source (original test JSONL),
--output, and repeated --baseline NAME=CSV. It requires the fixed 9015 cohort;
missing baseline predictions count as empty predictions.

Table 2 direct model: rescore_restored_joint_table.py accepts --predictions,
--source, --cohort paper_results/heterogeneous_cohort.csv, --output, --threshold .27.
Reference components, matching and the 9149-call budget retain the recorded protocol.
Baseline counts use reviewer_checks_20260914/run_checks.py with --root ASSETS,
--cache original_NPZ_cache, --chainsaw CSV, --merizo CSV, --m2-predictions JSONL_GZ,
--output. Its QuaSF row is the original cache control; the paper direct-model row
comes from the separate direct scorer, not that cache. The actual baseline is
M2 protein probabilities broadcast over external regions; no RegionMLP was used
for these reported counts. code/baselines/m2 contains its source model/trainer;
weights and archived exported probabilities are external assets.

Regional preference: quasf/paper_evaluate.py scores direct predictions;
code/shared_query_ablation holds the shared comparator. Both use observed labels
of matched regions and within-protein exclusive-label probability directions.

Structure figure: joint_function_query_20260924/build_structure_cards_v5.py,
audit_figure3_v5.py and multidomain_figure_manifest_v4.json. It requires the four
original structure-render PNGs and render-provenance JSONs under structures/rendered.
The region masks and labels are fixed; the manifest scores reflect the direct model.

paper_results contains current numerical summaries, baseline-count evidence and
the fixed heterogeneous cohort. Old exploratory and omitted ablation results are
recoverable from archive/pre-paper-only-20260930.
