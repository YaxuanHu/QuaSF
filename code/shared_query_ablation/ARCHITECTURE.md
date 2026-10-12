# Shared-query controlled comparison

This isolated experimental package preserves the flat training/evaluation interface
of the recorded 18-epoch shared-query comparison.
`dual_model.py` is the byte-preserved reference implementation. `model.py` extends
its module construction and auxiliary-query builders with a shared regular-query
forward path. Existing dataset, losses, Lightning orchestration and evaluator
interfaces remain unchanged. `train.py` remains the CLI adapter.

Public contract: DualQueryDecoupledModel(**reference_arguments), forward(features,
valid_mask, gt_spans=None, gt_func_targets=None, training=False) -> reference output
dictionary. The legacy exported class name is an evaluation compatibility alias;
the implementation is SharedQueryModel, recorded in the experiment manifest.

One six-layer decoder (concatenation of the reference's two three-layer stacks)
updates one regular-query set. Mask and function heads consume the exact same
final tensor. All six layers have independent parameters. The two stacks are
initialized exactly as in the seeded reference construction. GT-function and
denoising auxiliary groups use the same decoder, with bidirectional group
isolation; they never alter regular-query outputs. Existing losses, matching,
curriculum and encoder freeze remain unchanged. Fusion receives [z,z]. The
unused regular function-slot prototype is removed; report the parameter delta.

This compares shared versus specialized query pathways at matched total decoder
layer budget, not a pure parameter-sharing experiment or proof of gradient
conflict. Shared depth is six per task; dual depth is three per branch. Auxiliary
queries also traverse six rather than three layers. Document these consequences.

New checkpoint keys for the six-layer seg_decoder are intentionally incompatible
with dual checkpoints. Never warm-start from the trained reference. Existing
same-architecture checkpoints must strict-load with optimizer/scheduler intact.

User has authorized this bounded experiment. No additional interface approval
is needed. Source, metadata, configuration and selection are frozen before test.
Train seed42, 18 epochs, 4 GPUs, batch32/rank, accumulation1, full datasets.
Primary selection max validation tri score; final epoch sensitivity. No changing
budget or choosing metrics/checkpoints based on favorable test results.
