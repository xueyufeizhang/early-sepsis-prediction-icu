# Stage 4 训练断点续跑

LR、XGBoost、Random Forest 和 SVM 共用按“候选配置 × fold”保存的 checkpoint。
它记录已经完成的工作；不改变冻结的五折划分、不接触内部测试集评估，也不忽略训练失败。

SVM 的内部概率校准仍属于一个外层 fold，内部中断会重跑该未完成的外层 fold；
具体运行入口与耗时说明见 [SVM 训练说明](svm_training.md)。

## 开启方式

在原来的训练调用中增加 `checkpoint_dir` 即可。下面代码只能在存放受保护数据的受控环境运行；
`static` 和 `splits` 继续使用已有 Stage 2 特征与冻结的 Stage 3 划分，不需要重新生成。

```python
from src.config import DATA_PROCESSED
from src.models.classic import (
    train_logistic_regression,
    train_xgboost,
    train_random_forest,
    save_static_training_result,
)

lr_result = train_logistic_regression(
    static,
    splits,
    profile="tuning",
    checkpoint_dir=DATA_PROCESSED / "checkpoints" / "logistic_regression_tuning_v1",
    resume=True,
    progress_callback=print,
)
lr_artifacts = save_static_training_result(lr_result, artifact_suffix="tuning_v1")
```

`resume=True` 是默认值：新目录开始训练，已有且匹配的目录恢复训练。
不传 `checkpoint_dir` 或设为 `None` 时禁用 checkpoint，保留原来的行为。
`resume=False` 仅用于全新目录；它不是覆盖、清空或强制忽略旧记录的开关。
不同模型、不同实验必须使用不同目录，例如 `xgboost_tuning_v1`、`random_forest_tuning_v1`。

其他模型的训练入口同样支持这两个参数：

```python
xgb_result = train_xgboost(
    static, splits, profile="tuning", device="cuda",
    checkpoint_dir=DATA_PROCESSED / "checkpoints" / "xgboost_tuning_v1",
    resume=True, progress_callback=print,
)

rf_result = train_random_forest(
    static, splits, profile="tuning",
    checkpoint_dir=DATA_PROCESSED / "checkpoints" / "random_forest_tuning_v1",
    resume=True, progress_callback=print,
)
```

XGBoost 的 `device` 按实际运行环境选择；恢复时保持不变。RF 不传 `device`。
上面两个结果完成后仍需分别调用 `save_static_training_result`，使用本次实验的新后缀。

## 中断后怎么继续

1. 确认 checkpoint 目录及受保护的输入文件仍然存在。
2. 如果 kernel 已重启，重新执行导入与数据加载，加载原来的特征和冻结划分。
3. 保持相同代码、依赖版本、候选配置、设备和目录，重新执行相同训练调用。
4. 已完成的 fold 从 checkpoint 恢复；尚未完成的 fold 重新训练。
5. 全部候选完成后，照常选优、汇总 OOF 并得到最终结果，再保存正式产物。

最终赢家在完整 dev 上的 refit 也有 checkpoint。若该步骤已经完整保存，恢复时可以直接加载，
不必重新 refit；若是在 refit 中途崩溃，则重跑这一步，已完成的五折结果仍可复用。
恢复的粒度不是单棵树、单次 boosting 迭代或单个训练批次：正在执行且尚未保存的 fold 会重跑。

训练异常仍会抛出；先检查原因，再决定是否恢复。checkpoint 不会把失败候选当成成功结果。
如果训练已返回、只是之后保存或画图报错，且 kernel 仍在，可直接重试保存或画图。
注意：一次新的赋值调用失败时，同名变量可能仍指向上次实验的旧结果，不要误保存。

## 为什么有时会拒绝恢复

目录内的实验清单会严格校验开发集输入数据、完整冻结划分、候选参数、预处理配置、相关代码、
运行依赖版本和设备等信息。恢复不是按 `candidate` 名字相同就直接复用。

- 改数据或行顺序、重新划分、改候选集合/参数/随机种子：使用新目录。
- 改预处理或训练代码、升级相关依赖：使用新目录，不强行混用旧 fold。
- XGBoost 改 CPU/GPU 设备，或更换不兼容的运行版本：使用新目录。
- 记录损坏或不完整：检查报错，不手改清单、不删除校验字段绕过检查。

发生不匹配时，保留旧目录并为新实验取新名字；不要清空旧 checkpoint 来掩盖不匹配。
同一个 checkpoint 目录只允许一个训练任务写入，不要从多个 notebook 或进程并发使用。

## checkpoint 与正式产物的区别

checkpoint 用于恢复训练进度；`save_static_training_result` 用于导出最终模型、OOF 和结果表。
两者不能互相替代。看到一个 checkpoint 文件，不代表整次实验或全部正式产物已经完成。
旧版本没有启用 checkpoint 的运行，无法事后恢复当时丢失的中间进度。

## 文件保留与合规

checkpoint 是磁盘文件，不是 Kaggle/Colab 云备份。kernel 重启后能否恢复，取决于文件是否仍在；
会话结束、环境被回收或目录丢失后，本功能不能凭空找回文件。请在受控环境内确认合规的持久化方式。

包含患者级 OOF、标签或拟合模型的 checkpoint 都属于受保护产物。
统一放在 `DATA_PROCESSED / "checkpoints"` 下；不要提交 Git、发布为公开 notebook/output/dataset，
也不要发送给第三方 API 或 LLM。不要修改 `.gitignore` 来追踪这些文件。
只从自己受控、可信的目录恢复 `joblib` 模型；加载不可信的序列化模型可能执行任意代码。
