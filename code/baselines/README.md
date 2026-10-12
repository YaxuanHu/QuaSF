# Paper baselines

Chainsaw official inference: run_official_chainsaw.py + chainsaw-main/.
Merizo official inference: run_official_merizo.py + merizo/predict.py + merizo/model/.
EguchiCNN: EguchiCNN/ model, trainer, cached-feature converters and evaluators.
SWORD and UniDoc: sword/ and unidoc/ external-method evaluation wrappers.
evaluate_metrics.py, prepare_baseline_csvs.py and feature-preparation scripts are
shared dependencies; weights and prepared feature files are external assets.

Table 2 functional baseline uses M2 protein-level probabilities, broadcast over
Chainsaw/Merizo regions. m2/ contains the source model, data, losses and trainer.
m2/config.json records the reported experiment's configuration; train.py generic
defaults are not that experiment's full invocation. Recorded probability exports
are external inputs to analysis/reviewer_checks_20260914/run_checks.py.
No RegionMLP classifier is part of the Table 2 broadcast comparison. The later regional-preference comparison uses the separate Region MLP under analysis/region_mlp_trial_20261004/.
