# Paper-only release, 2026-09-30

Retains the selected pretrained direct model and only paper-related baseline,
comparison, metric and figure source. Residual repair, RegionMLP and omitted
experiments are removed. Direct model layers and region helpers are extracted
with unchanged state-dict names and numerical operations.

Remote full-source backup: archive/pre-paper-only-20260930,
commit 2b7494240a03395273c4cbd4b7199699817ff51a. docs/CLEANUP.json lists deleted files.
The data/checkpoint assets remain external. Source archive includes retained paper
experiments; it contains no data, weights, caches or virtual environment.
Build with python scripts/build_release.py. See RELEASE_FILES.json for file hashes,
RELEASE_PROVENANCE.json for source integrity and RELEASE_VERIFICATION.json for
independent old/new selected-checkpoint forward parity.
