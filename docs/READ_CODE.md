# 主模型阅读顺序

完整模型：quasf/model.py。
1. BatchedJointFunctionQuery.forward：分割、区域池化、功能 query、最终输出。
2. DirectFunctionQuery：独立功能模块，没有 RegionMLP，也不继承残差修补类。
3. BatchedJointFunctionQuery.decode_regions / region_features：区域后处理和池化。
4. DualQueryDecoupledModel：真实 checkpoint 对应的分割骨干及蛋白辅助表示。

训练：pretrain.py → paper_train.py → train.py 中的 JointCurriculum。
评估：paper_evaluate.py。论文比较方法：code/baselines/ 和 code/shared_query_ablation/。
表格、结构图：analysis/README.md。

服务器主模型：
/data/huyaxuan/code/region_function_multimodal/release/QuaSF_20260930/quasf/model.py
