# WEEK 13 当前阶段进度

截至 2026-11-06，已完成实验方案统一、两个神经网络的首次训练，以及小规模调参与三种子验证。验证误差分析也已完成；测试集尚未评价，英文一页报告和代码提交材料已整理，PPT 按安排留待后续。

## 已完成的公共数据流程

- 数据：2021-07-01 至 2026-07-01 的成都 PM2.5 分项 AQI，仅使用 pm25。
- 1,827 个日历日、1,821 个日期记录、1,817 天有效值；6 天缺行、4 天值为空。
- 保留日历，跳过不完整窗口，共 1,782 个连续七天输入及次日标签样本。
- 按完整日历 70/15/15 边界、根据目标日期划分：训练 1,261、验证 265、测试 256。
- 标准化参数只在训练时段拟合；逐日滚动单步预测可使用预测时已观测的历史实际值。

| 分组 | 首个有效标签 | 最后有效标签 | 样本数 |
|---|---|---|---:|
| 训练 | 2021-07-08 | 2024-12-29 | 1,261 |
| 验证 | 2025-01-08 | 2025-09-29 | 265 |
| 测试（尚未评价） | 2025-09-30 | 2026-07-01 | 256 |

## 六项任务状态

1. 已完成：固定实验方案并统一 Decoder-only 文档。详见 experiment_protocol.md。
2. 已完成：Decoder-only Transformer 的单配置训练、保存加载及预测；用户也已用相同种子亲自复现指标。
3. 已完成：小型 LSTM 的单配置训练、保存加载及预测；核对四模型验证日期和真实值一致。
4. 已完成：三个配置/模型的少量调参，选定配置的 42、43、44 三种子比较及正确性检查。
5. 已完成：分组误差、最大误差案例、训练曲线及实际问题整理，见 stage5_error_analysis.md。
6. 本次已整理代码、复现说明与英文一页 Word 报告；PPT 按用户安排留待后续。

## 任务 2 和任务 3 的首次验证结果

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 20.44 | 27.10 | 0.416 |
| Linear Regression | 18.89 | 25.13 | 0.498 |
| LSTM（seed=42） | 18.88 | 24.78 | 0.512 |
| Decoder-only Transformer（seed=42） | 18.76 | 24.34 | 0.529 |

MAE/RMSE 单位为 AQI 指数点，R² 无单位。本轮 Transformer 验证 RMSE 最低；两个神经网络各只有一个配置、一个种子的结果，不能据此声称稳定优于其他模型。测试集未评价。

Transformer：1 层、32 维、2 heads、前馈维度 64、dropout=0.1，共 8,929 参数。39 轮后提前停止，最佳第 19 轮。验证 MAE=18.7645、RMSE=24.3379、R²=0.5293。

LSTM：单向、1 层、32 隐藏单元，最后隐藏状态接 dropout=0.1 和回归头，共 4,513 参数。31 轮后提前停止，最佳第 11 轮。验证 MAE=18.8813、RMSE=24.7814、R²=0.5120。

两者使用 seed=42、CPU、AdamW、lr=0.001、batch=32、weight_decay=0.0001、梯度裁剪 1.0、最多 150 轮及 patience=20。按验证 RMSE 保存最佳模型，并还原到 AQI 单位评价。共同训练逻辑在 sequence_training.py；原 Transformer 命令和保存权重仍可使用。

## 已完成检查

任务 3 时十项单元检查通过：原有四项数据流程检查、三项 Transformer 检查、三项 LSTM 检查。LSTM 检查包括样本间不共享隐藏状态、保存重载及反标准化、非法历史输入拒绝。

实际结果另验证：四模型的 265 个验证目标日期及真实值一致；LSTM 与 Transformer 的 scaler、日期划分和数据指纹一致；最佳轮次与训练记录一致；LSTM 最佳模型重载后可复现预测，命令行预测入口可运行。将训练逻辑提取到公共模块后重跑 Transformer，验证指标及全部逐日预测与原结果完全一致。

## 文件与复现

按 README.md 操作。数据与结果仅保存在本地，不上传公开仓库。

- baseline：pipeline.py，结果 results/baselines。
- Transformer：transformer_model.py、train_transformer.py、predict_transformer.py；结果 results/transformer/seed42，用户复现位于 seed42_my_run。
- LSTM：lstm_model.py、train_lstm.py、predict_lstm.py；结果 results/lstm/seed42。
- 共用训练流程：sequence_training.py。
- 四模型初步指标表：results/lstm/seed42/model_comparison.csv。

## 可用于汇报的准确措辞

We have implemented a chronological preprocessing and validation pipeline and trained a lightweight causal decoder-only Transformer and a small LSTM. Both use the same seven-day inputs, training-only scaling and validation dates as the comparison protocol. We have initial validation results for all four models. We completed a small validation-based configuration search, three-seed comparisons and validation error analysis. The held-out test set has not yet been evaluated.

## 任务 4 已完成

完整结果与选择规则见 [stage4_results.md](stage4_results.md)。共十次训练：Transformer 选原始配置，LSTM 选学习率 0.0003。选定配置的三种子均值 ± 样本标准差：

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| persistence | 20.44 | 27.10 | 0.416 |
| linear_regression | 18.89 | 25.13 | 0.498 |
| transformer | 18.66 ± 0.09 | 24.31 ± 0.02 | 0.530 ± 0.001 |
| lstm | 18.89 ± 0.05 | 24.74 ± 0.03 | 0.514 ± 0.001 |

均值与标准差为各次指标汇总，不是 ensemble。十四项单元检查通过，十次实验的重载、日期、标准化和指标核对通过，--resume 已验证。三种子不是最终测试或显著性证明。任务 5 已在后续完成，接下来为任务 6。

## 任务 5 已完成

验证诊断覆盖同一批 265 日。大幅变化阈值为训练绝对日变化第 90 百分位，约 44.8 AQI 点。18 个大幅变化日，Transformer 三种子平均 MAE=52.31；其余 247 日为 16.21。普通日与 6 个大幅上升日，线性回归 MAE 略低，不能将 Transformer 的总体优势外推到所有场景。

seed=42 最大误差为 2025-01-29，真实值 282，Transformer 预测 132.52，绝对误差 149.48。四模型均未预测出这次峰值。训练曲线显示最佳轮后训练损失继续下降但验证指标没有改善，保留最佳权重有必要。

十六项检查通过，四张分析图已查看，分组与月份计数、总体指标和输入指纹已独立核对。结果在 results/stage5，中文素材见 stage5_error_analysis.md。没有重新训练、改变已选配置或评价测试集。任务 6 的报告和代码整理已推进，PPT 按安排留待后续。

## 任务 6 本次交付

英文一页报告保存在 reports/Chengdu_PM25_Week13_Report.docx，已在 WPS 中核对为一页。报告指标逐项核对任务 4 的均值及样本标准差，明确保留测试集状态。完整复现步骤及提交范围见 reproduction.md。本地代码备份包排除原始数据、权重、results、缓存和临时文件。PPT 按当前安排留待后续，最终测试仍未评价。
