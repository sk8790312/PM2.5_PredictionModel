# PM2.5_PredictionModel

成都 PM2.5 AQI 次日预测课程项目。当前代码阶段使用 2021-07-01 至 2026-07-01（含首尾两日）的历史数据，以过去连续七天预测次日。

当前仓库仅覆盖本阶段的数据处理、简单 baseline 和初步验证。后续 Transformer、LSTM 训练及最终测试待下一阶段再推进。

## 当前阶段正在进行

1. 读取数据Excel，检查日期、空值及重复记录。
2. 补齐日历轴，以识别完全缺行的日期；不填补污染物数值。
3. 构造连续七天输入和次日标签，跳过含缺失的窗口。
4. 按目标日期分为训练、验证、测试三个时段（约 70% / 15% / 15%）。
5. 训练线性回归，并与 persistence（明天等于今天）比较；只报告验证集成绩。

**数据含义：** AQICN 导出的 `pm25` 是 PM2.5 单项 AQI，不是 μg/m³ 浓度或整体 AQI。此次代码工作的目标名称据此记录；无需回头修改已经提交的 proposal。文件未提供足够站点及日汇总元数据，这些限制需在阶段汇报中说明。

## 第一次运行：编辑器（如VS Code） + Anaconda（Windows）

Anaconda 管理 Python 环境；编辑器 用来打开项目、阅读代码和查看结果。当前阶段使用普通 CPU 即可。

### 1. 把 main 分支下载到电脑

打开本仓库的 main 分支，点击 **Code → Download ZIP**，解压到你自己选择的文件夹。后面所说的“项目目录”，是包含 `pipeline.py` 和 `requirements.txt` 的那一层文件夹。

如果电脑已经安装 Git，也可以在保存项目的位置打开终端执行：

```text
git clone --branch main https://github.com/sk8790312/PM2.5_PredictionModel.git
```

### 2. 用 Anaconda 创建项目环境

从 Windows 开始菜单打开 **Anaconda Prompt**，执行：

```text
conda create -n pm25 python=3.12 pip -y
conda activate pm25
```

环境只需创建一次；以后使用时执行 `conda activate pm25` 即可。如果已经有名为 `pm25` 的环境，先确认它是本项目环境，不要重复创建。

进入项目目录。将下面的示例路径替换成你实际解压或克隆的文件夹路径，并保留双引号：

```text
cd /d "D:\Projects\PM2.5_PredictionModel-main"
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
```

最后一条会输出本环境的 Python 路径，下一步选择解释器时可以使用。上面的 `cd /d` 是 **Anaconda Prompt** 中的命令，不要直接复制到 PowerShell。

### 3. 在 编辑器 中打开项目

1. 打开，选择 **File → Open Folder**，打开上一步的项目目录。
2. 确认已安装并启用 Python 扩展。
3. 按 **Ctrl+Shift+P**，搜索 **Python: Select Interpreter**，选择名为 `pm25` 的 Conda 环境。
4. 如果列表里没有该环境，选择手动输入解释器路径，填写上一步输出的 Python 路径。

解释器选择控制编辑器使用哪个 Python；终端也需要使用同一个环境。

### 4. 放好本地 Excel 数据

把自己保存的整理后 Excel 复制到项目的 `data` 文件夹中，文件名为：

```text
data/chengdu_20210701-20260701_sorted.xlsx
```

文件只保存在自己电脑上，无需在网页上传。`.gitignore` 已排除 Excel 数据文件。

### 5. 运行当前阶段代码

在 编辑器 中选择 **Terminal → New Terminal**，确认终端位于项目目录，并激活环境：

```text
conda activate pm25
python -c "import sys; print(sys.executable)"
python pipeline.py --data "data/chengdu_20210701-20260701_sorted.xlsx"
```

确认 Python 路径属于 `pm25` 环境。如果编辑器终端提示找不到 `conda` 或无法激活，直接回到第 2 步已经激活环境、进入项目目录的 **Anaconda Prompt**，运行最后一条命令即可；仍然可以用 编辑器 阅读代码和查看结果。

程序会在 `results/baselines/` 下生成数据检查、验证集指标和图片。可以在编辑器文件列表中打开 `summary.json`、`data_overview.png` 和 `validation_forecasts.png`。

### 6. 检查流程

在同一个已激活环境的终端中执行：

```text
python -m unittest discover -s tests -v
```

正常时四项检查通过。当前阶段仅运行数据处理和两个简单 baseline，测试集评价及 Transformer、LSTM 训练留待后续阶段。

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
