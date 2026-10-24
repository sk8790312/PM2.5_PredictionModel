# 当前阶段代码进展

本记录来自已整理的 Excel 在本地的一次实际运行；不是 Transformer 的结果。

## 已完成

- 以 2021-07-01 至 2026-07-01 的 PM2.5 AQI 为研究对象，仅使用历史 pm25。
- 审计并保留缺失日历：1,827 天，1,821 个日期记录，1,817 天有效值；6 天缺行、4 天值为空。
- 跳过不完整窗口，生成 1,782 个七天输入、次日标签样本。
- 按完整日历约 70/15/15 划分；训练/验证/测试有效样本分别 1,261 / 265 / 256。
- 拟合线性回归及其训练集标准化，与 persistence 比较。
- 四项检查通过：缺失不能跨越、标签日期划分、空格与缺行区分、重复日期拒绝。

有效标签日期范围：

| 分组 | 首个有效标签 | 最后有效标签 | 样本数 |
|---|---|---|---:|
| 训练 | 2021-07-08 | 2024-12-29 | 1,261 |
| 验证 | 2025-01-08 | 2025-09-29 | 265 |
| 测试（尚未评价） | 2025-09-30 | 2026-07-01 | 256 |

验证集初步结果（不代表最终测试成绩）：

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 20.44 | 27.10 | 0.416 |
| Linear Regression | 18.89 | 25.13 | 0.498 |

MAE、RMSE 单位为 AQI 指数点。该次验证结果中线性回归优于 persistence，尚不能推断 Transformer 或 LSTM 的表现。

## 下一步

解释并复现这条流程，然后在相同样本和划分上加入小型 encoder-only Transformer 与 LSTM。使用验证集选择配置，最终测试留到方案确定后。不要把计划中的模型说成已经完成。

## 可用于汇报的准确措辞

We have implemented a Python preprocessing and evaluation pipeline. We constructed 1,782 complete seven-day input windows and split them chronologically into training, validation, and test sets. We obtained initial validation results for persistence and linear regression. Transformer and LSTM training are planned next; the test set has not yet been evaluated.

## 复现

详见 README.md。实际运行的完整指标、环境版本与图表在本地 results/baselines/ 下，这些包含观测数据的文件不会提交到公开仓库。
