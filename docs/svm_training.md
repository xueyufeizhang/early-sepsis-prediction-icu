# Stage 4：SVM 训练与概率校准

SVM 已接入 `src/models/classic.py` 的公共训练框架。继续使用已有 Stage-2 特征和冻结的
Stage-3 划分，不需要重新运行数据准备或拆分。真实数据训练只在用户受控的 Kaggle/Colab
环境运行；本地仅测试合成数据。

## 代码怎么串起来

- `svm_candidates()`：生成 kernel、C、gamma 与不平衡策略的组合，所有 SVM 开启数值标准化。
- `build_svm()`：构建 `SVC`，在当前实际训练子集内计算出的权重传入 `class_weight`。
- `FoldLocalStaticClassifier`：每次 `fit()` 都为收到的原始训练子集重新构建预处理与采样流程。
- `_fit_candidate_predictor()`：普通模型直接训练；SVM 使用患者分组的三折 sigmoid 概率校准。
- `train_svm()`：复用候选比较、外层 OOF、checkpoint 和最佳模型的完整开发集重训。
- `run_svm_stage4()`：加载、训练、保存的一站式入口；notebook 分开调用训练和保存方便检查。

### 为什么有内层三折

现有外层五折负责产生用于比较候选的开发集 OOF 预测。对每个外层训练集，内部再按
`subject_id` 划分三折：用内层训练部分训练 SVM，为内层验证部分输出 decision score。
校准器利用这些折外分数和真实标签，学习 sigmoid 的“分数 → 概率”映射。

内层每次拟合都重新计算训练权重、拟合预处理，必要时仅对训练行执行 SMOTENC。
内层验证行不重采样，保持原始类别分布。`ensemble=False` 随后在整个外层训练集重训基础
SVM，配合已学到的校准器预测外层验证集。最终完整 dev 重训也使用相同的内部校准步骤。

内层折用于**校准概率，不是另一次超参数搜索**。外层验证集和内部测试集都不参与该次校准。
候选仍按公共框架的开发集平均折 AUROC、AUPRC 等规则选优；用同一开发集选择候选后的成绩
属于开发期结果，不能代替后续封存测试集评估。Stage 5 才选择 F1 阈值和进行最终评估。

## 运行顺序

更新代码后重启 kernel，重新执行 imports 和原有数据加载，然后运行 notebook 第 6 节。
无需为了运行 SVM 重跑已有 LR/XGBoost 实验。

1. 检查候选表。`smoke` 为 3 个；默认 `tuning` 为 36 个；可选 `screening` 为 6 个。
2. 把 `RUN_SVM_SMOKE` 改为 `True`。smoke 显式使用 `checkpoint_dir=None`，不写断点。
3. 核对三种策略的汇总指标、OOF 完整性和耗时，再打开 `RUN_SVM_TUNING`。
4. 调参使用 `DATA_PROCESSED / "checkpoints" / "svm_grouped_calibration_v1"`
   和 `resume=True`，完成后另行保存模型、OOF 和汇总结果。

smoke 固定 RBF、`C=1`、`gamma="scale"`，对比 baseline、class weight 和 SMOTENC 0.25。
调参比较 linear 的三个 C 值（0.1、1、10）与 RBF 的九个 C/gamma 组合
（C 同前，gamma 为 `"scale"`、0.01、0.1），共 12 种模型配置，再分别配合 baseline、
class weight、SMOTENC 0.10，共 36 个候选。linear 不重复遍历无效的 gamma 选项。

`SVC` 使用 CPU，不传 `device`、`n_jobs` 或 `probability=True`。外部校准器负责概率输出。
保存的最终模型已包含预处理、基础 SVM 与校准器，加载后可直接调用 `predict_proba()`；
不要另外对输入拟合新的 scaler 或另做一次 SMOTENC。

### 耗时与断点

每个外层候选/fold 包含三次内层训练和一次完整外层训练集重训，即四次基础 SVM 拟合。
无缓存且成功完成时，smoke 总计 `3 × 5 × 4 + 4 = 64` 次，默认 tuning 总计
`36 × 5 × 4 + 4 = 724` 次，末尾的 4 次是全局赢家在完整 dev 上的最终校准与重训。
候选数不是实际基础模型拟合次数；先通过 smoke 判断当前 CPU 的可承受耗时。

checkpoint 粒度仍为“候选 × 外层 fold”。若在内层校准中途失败，重跑那个未完成的外层 fold；
已经保存的其他外层 fold 可复用。`resume=False` 不等于禁用 checkpoint，也不会覆盖旧目录。

本次修改改变了共享 `classic.py` 的代码指纹，因此旧版未完成的 LR/XGBoost/RF checkpoint
也可能拒绝在新代码下恢复。需要继续旧实验时使用它原来匹配的代码与依赖；开始新实验时用新目录。
不要手改 manifest 来跳过校验。已有正式结果不会被自动删除，也不需要重新生成 Stage-3 划分。
后续改候选、代码或依赖时，目录和正式结果后缀都应升级版本，避免覆盖本次产物。

断点文件只有在受控环境的磁盘文件仍存在时才能恢复；它不是会话回收后的自动云备份。
所有患者级 OOF、checkpoint 与拟合模型都是受保护衍生物，不得提交 Git、公开发布或发送给
第三方 API/LLM。只分享必要的聚合指标，且只加载自己可信环境中的序列化模型。

详细通用规则见 [训练断点说明](training_checkpoints.md)。
