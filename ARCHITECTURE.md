# QuaSF publication architecture

The current model is BatchedJointFunctionQuery(segmenter, DirectFunctionQuery()).
ESM residue features -> segmentation masks -> decoded final regions -> raw-ESM
mean/max pooling -> own-region attention -> inter-region attention -> direct logits.
DirectFunctionQuery independently defines its layers; no MLP baseline or residual
repair class is instantiated or retained. Its transformer FFN remains necessary.

DualQueryDecoupledModel retains the exact pretrained backbone parameter namespace
and its functional path because the detached protein residual uses its outputs.
Anchor and fusion projection layers are not the removed RegionMLP classifier.
Final regional outputs are region_masks[B,R,L], region_logits[B,R,256],
region_valid[B,R] and region_source_query_ids[B,R]. Compatibility func_logits[B,Q,256]
aggregate final regions by source query. delta_head and query_delta retain their
checkpoint names but represent complete direct class logits.

Scientific computation lives in quasf/model.py, losses.py and metrics.py.
Data/checkpoint adapters live in data.py, region_cache.py and paper_train.py.
Root and installed CLIs call pretrain.py, paper_train.py and paper_evaluate.py.
train.py contains only the current JointCurriculum; evaluate.py contains scoring helpers.
Models and losses use PyTorch directly, preserving the scientific package boundary.

Paper baselines and the shared-query comparator are separate under code/.
Paper table/figure adapters are under analysis/ and take assets as inputs.
No quasf module imports experimental code. Delete omitted experiments rather than
keeping import-compatible historical implementations in the current package.

Cleanup preserves parameter names, initialization order, mathematical operations,
losses, thresholds and checkpoint identities. Unused inheritance is flattened.
Archive-forward and gradient parity plus strict real-checkpoint restoration verify
compatibility. No checkpoint conversion, new training or full-test rerun occurs.
The remote backup tag is archive/pre-paper-only-20260930.

2026-10-12 packaging update: Region MLP is restored only under analysis as an independent paper comparator. Core computation, public CLIs and checkpoint namespaces are unchanged. Release assets remain outside the Python package. See docs/PAPER_CODE_MAP.md.
