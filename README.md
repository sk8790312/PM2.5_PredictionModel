# PM2.5_PredictionModel

成都 PM2.5 AQI 次日预测课程项目。当前代码阶段使用 2021-07-01 至 2026-07-01（含首尾）的历史数据，以过去连续七天预测次日。

当前仓库仅覆盖本阶段的数据处理、简单 baseline 和初步验证。后续 Transformer、LSTM 训练及最终测试待下一阶段再推进。

## 当前阶段做什么

1. 读取你自己的 Excel，检查日期、空值及重复记录。
2. 补齐日历轴，以识别完全缺行的日期；不填补污染物数值。
3. 构造连续七天输入和次日标签，跳过含缺失的窗口。
4. 按目标日期分为训练、验证、测试三个时段（约 70% / 15% / 15%）。
5. 训练线性回归，并与 persistence（明天等于今天）比较；只报告验证集成绩。

**数据含义：** AQICN 导出的 `pm25` 是 PM2.5 单项 AQI，不是 μg/m³ 浓度或整体 AQI。此次代码工作的目标名称据此记录；无需回头修改已经提交的 proposal。文件未提供足够站点及日汇总元数据，这些限制需在阶段汇报中说明。

## 第一次运行：Google Colab

新建 Colab 笔记本，逐个执行下列单元格。代码从 main 分支读取。

```python
!git clone --branch main https://github.com/sk8790312/PM2.5_PredictionModel.git
%cd PM2.5_PredictionModel
!pip install -r requirements.txt
```

上传已整理好的 Excel（自己保留，不上传 GitHub）：

```python
from google.colab import files
uploaded = files.upload()
```

开始运行，文件名与实际上传的名字一致：

```python
!python pipeline.py --data "chengdu_20210701-20260701_sorted.xlsx"
```

查看结果：

```python
import json
from IPython.display import display, Image
with open('results/baselines/summary.json', encoding='utf-8') as f:
    summary = json.load(f)
print(summary['splits'])
print(summary['validation_metrics'])
display(Image(filename='results/baselines/data_overview.png'))
display(Image(filename='results/baselines/validation_forecasts.png'))
```

Colab 运行结束前下载结果：

```python
import shutil
shutil.make_archive('/content/pm25_results', 'zip', 'results/baselines')
files.download('/content/pm25_results.zip')
```

## 在自己电脑运行

使用 Python 3.10 或更高版本，在项目目录运行：

```text
python -m pip install -r requirements.txt
python pipeline.py --data "你的Excel完整路径"
python -m unittest discover -s tests -v
```

## 输出分别说明什么

| 文件 | 用途 |
|---|---|
| `summary.json` | 数据完整性、样本数量、划分日期、验证集 MAE/RMSE/R²、实际运行环境版本 |
| `missing_dates.csv` | 每个缺失日期及原因（完全缺行或 PM2.5 为空） |
| `daily_calendar.csv` | 完整日历，缺失值仍为空，便于复核 |
| `samples.csv` | 七个历史输入、标签及所属分组 |
| `validation_predictions.csv` | 验证集真实值与两个 baseline 的预测 |
| `data_overview.png` | 历史 PM2.5 AQI 趋势；缺失位置断线 |
| `validation_forecasts.png` | 验证集前 90 个有效预测日期的对比 |

`lag_7` 是目标日前第七天，`lag_1` 是目标日前一天。比如 7 月 1–7 日输入，预测 7 月 8 日。

切分依据是完整日历的固定日期边界，标签互不重叠。验证和测试预测可以使用预测时已经观测到的前一天实际值，这是逐日滚动的单步预测，不是一次性预测未来一整年。标准化和线性回归只使用训练集拟合。

MAE/RMSE 单位为 AQI 指数点，R² 无单位。较低的 MAE/RMSE 通常更好，R² 越高通常更好，也可能为负。

本阶段仅核查测试集数量，不提供测试集评价入口；最终测试留待后续阶段。各次运行建议使用独立 `--output` 目录，避免混淆实验记录。

## 还未完成的工作

- 小型 encoder-only Transformer：数值投影、位置编码、1–2 个 Encoder 层和回归头。
- 小型 LSTM，作为序列模型 baseline。
- 依据验证集选择超参数，再进行最终测试及多次随机种子实验。
- 如加入其他污染物，重新检查特征缺失，并使用相同评价日期比较。

以上为下一步计划，不能写成已训练完成。当前阶段已跑通的范围仅包括数据流程、persistence 和线性回归。

## 来源与数据保存

来源：https://aqicn.org/historical/cn/#city:chengdu 。感谢 WAQI 和原始环境监测机构；数据未经 WAQI 充分验证。详细含义及限制见 `data/README.md`。

仓库不包含原始 Excel 或含真实观测的结果文件，`.gitignore` 已将它们排除。每位成员使用自己的下载文件复现，不向公开仓库提交数据。Python 依赖范围见 `requirements.txt`；当前实际运行版本记录在每次输出的 `summary.json` 中。
