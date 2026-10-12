# Shared-query paper comparator

model.py defines a single regular-query state for masks and functions. Its six
decoder layers match the dual backbone's total decoder-layer count. dual_model.py,
dataset.py, loss.py, lightning_module.py and train.py retain checkpoint/training
dependencies. Invoke train.py with the recorded source hyperparameters rather
than generic defaults. Cache its selected checkpoint with
code/segmentation_decode_ablation/cache_predictions.py using an explicit --config.
Score regional preference with analysis/regional_preference.py. Weights, cache
and validation-selected identities are external experiment assets.
