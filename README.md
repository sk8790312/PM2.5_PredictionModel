# PM2.5_PredictionModel

成都 PM2.5 AQI 次日预测课程项目。当前代码阶段使用 2021-07-01 至 2026-07-01（含首尾两日）的历史数据，以过去连续七天预测次日。

当前已实现数据处理、两个简单 baseline，以及轻量级 Decoder-only Transformer 和小型 LSTM 的第一轮训练、验证与保存加载。WEEK 13 任务 1–5 已完成，包括小规模调参、三种子验证及误差分析。一页报告和代码提交材料已整理，PPT 按安排留待后续；测试集尚未评价。实验约定见 [docs/experiment_protocol.md](docs/experiment_protocol.md)。

## 已完成的数据与 baseline 流程

1. 读取数据Excel，检查日期、空值及重复记录。
2. 补齐日历轴，以识别完全缺行的日期；不填补污染物数值。
3. 构造连续七天输入和次日标签，跳过含缺失的窗口。
4. 按目标日期分为训练、验证、测试三个时段（约 70% / 15% / 15%）。
5. 训练线性回归，并与 persistence（明天等于今天）比较；只报告验证集成绩。

**数据含义：** AQICN 导出的 `pm25` 是 PM2.5 单项 AQI，不是 μg/m³ 浓度或整体 AQI。此次代码工作的目标名称据此记录；无需回头修改已经提交的 proposal。文件未提供足够站点及日汇总元数据，这些限制需在阶段汇报中说明。

## 第一次运行：Cursor／VS Code + Anaconda（Windows）

Anaconda 管理 Python 环境；Cursor 或 VS Code 用来打开项目、阅读代码和查看结果。当前阶段使用普通 CPU 即可。

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

### 3. 在 Cursor／VS Code 中打开项目

1. 打开 Cursor 或 VS Code，选择 **File → Open Folder**，打开上一步的项目目录。
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

在 Cursor／VS Code 中选择 **Terminal → New Terminal**，确认终端位于项目目录，并激活环境：

```text
conda activate pm25
python -c "import sys; print(sys.executable)"
python pipeline.py --data "data/chengdu_20210701-20260701_sorted.xlsx"
```

确认 Python 路径属于 `pm25` 环境。如果编辑器终端提示找不到 `conda` 或无法激活，直接回到第 2 步已经激活环境、进入项目目录的 **Anaconda Prompt**，运行最后一条命令即可；仍然可以用 Cursor 阅读代码和查看结果。

程序会在 `results/baselines/` 下生成数据检查、验证集指标和图片。可以在编辑器文件列表中打开 `summary.json`、`data_overview.png` 和 `validation_forecasts.png`。不需要 Colab 的上传、下载代码。

### 6. 检查流程

在同一个已激活环境的终端中执行：

```text
python -m unittest discover -s tests -v
```

正常时十六项检查通过，包括四项数据流程、三项 Transformer、三项 LSTM、四项实验协议和两项误差分析检查。最终测试留待后续阶段。

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

## 任务 2 Transformer 运行步骤

确保终端在项目目录，使用 pm25 环境。先安装更新后的依赖：

```text
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.__file__)"
```

已完成的首次训练结果在 `results/transformer/seed42/`。重新训练时使用新的输出目录，程序会拒绝覆盖已有实验：

```text
python train_transformer.py --data "data/chengdu_20210701-20260701_sorted.xlsx" --output "results/transformer/seed42_rerun"
```

训练入口使用训练时段有效 AQI 拟合一个共同的均值和标准差，转换所有历史位置和训练标签；验证与预测复用这两个参数，指标还原到 AQI 单位。它与线性回归按 lag 分列标准化的方式不同，两者均仅使用训练数据拟合预处理。

模型包含 1 层因果自注意力、32 维表示、2 个注意力头、64 维前馈层及回归头，共 8,929 个参数。数值投影和可学习位置表示后，以显式 causal mask 限制注意力，再取最后位置预测一个数值。代码使用 `TransformerEncoderLayer` 作为仅含自注意力的构件，实际结构由因果掩码约束；不含 Encoder memory 或 Cross-Attention。

默认 CPU、seed=42、AdamW、学习率 0.001、batch_size=32、最多 150 轮、patience=20，按验证 RMSE 保存最佳模型。训练与验证损失均在 eval 模式按每轮结束后的模型计算，以便比较；纵轴为标准化后的 MSE，不是 AQI 单位的 RMSE。

加载模型并预测，以下七个数值仅演示命令格式，实际使用须填写连续七个已观测日期的 AQI，按从早到晚排列：

```text
python predict_transformer.py --checkpoint "results/transformer/seed42/best_model.pt" --history 50 60 70 80 90 100 110
```

程序无法从七个裸数值自动验证日期是否连续，使用者需确认。预测输出是下一天 PM2.5 分项 AQI；不将负预测自动截断，以免隐藏模型行为。

任务 2 首轮结果：验证 MAE=18.7645、RMSE=24.3379、R²=0.5293；第 39 轮提前停止，第 19 轮为最佳模型。这是 seed=42 的单次结果；任务 4 的调参与三种子汇总另见下文。测试集未评价。

输出目录包含：`best_model.pt`（权重、结构及标准化参数）、`summary.json`（配置和指标）、`training_history.csv`、`validation_predictions.csv`、`loss_curve.png`、`validation_forecasts.png`。模型和结果位于被忽略的 results 目录，不随代码自动上传。

实际使用 PyTorch 2.7.1+cpu。其他电脑可先安装官方 CPU 版本，再安装其余依赖：

```text
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## 任务 3 LSTM 运行步骤

继续使用已有 pm25 环境，无需安装新依赖。在项目目录执行，使用新的输出目录保留首次结果：

```text
python train_lstm.py --data "data/chengdu_20210701-20260701_sorted.xlsx" --output "results/lstm/seed42_my_run"
```

已完成的首次结果在 `results/lstm/seed42/`。默认单向 LSTM，1 层、32 个隐藏单元；最后隐藏状态经过 dropout=0.1 和线性回归头输出次日 AQI，共 4,513 个参数。每个七天样本独立初始化隐藏状态，不在样本间传递记忆。输入形状为 [batch, 7, 1]，按最早到最近日期排列。

Transformer 和 LSTM 的训练逻辑统一放在 `sequence_training.py`，`train_transformer.py` 与 `train_lstm.py` 保留各自的运行命令。它们使用相同训练时段拟合的 AQI 标准化参数、日期划分、AdamW 设置、MSE 损失、提前停止规则和验证 RMSE 选择标准；各自单独训练和保存模型。为验证整理训练代码没有改变已有实验，已重跑 Transformer 并确认其验证指标及全部逐日预测与原结果完全一致。

LSTM 首轮 seed=42，CPU，最多 150 轮，patience=20；实际运行 31 轮，最佳第 11 轮，验证 MAE=18.8813、RMSE=24.7814、R²=0.5120。结果目录包含最佳权重、汇总、训练记录、验证预测、两张图；本次另保存四模型指标表 `model_comparison.csv` 和检查记录 `verification.json`。这是任务 3 的单配置历史结果；已完成的任务 4 调参与稳定性实验见下文。测试集未评价。

加载预测，以下数字仅演示格式，实际输入须是连续七个已观测日期的 AQI：

```text
python predict_lstm.py --checkpoint "results/lstm/seed42/best_model.pt" --history 50 60 70 80 90 100 110
```

训练程序拒绝覆盖非空结果目录。如果 `seed42_my_run` 已存在且有文件，请使用新的目录名。所有预测结果均还原到 AQI 单位；不把 R² 称为预测准确率。

四模型在相同 265 个验证日期上的初步结果：

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 20.44 | 27.10 | 0.416 |
| Linear Regression | 18.89 | 25.13 | 0.498 |
| LSTM（seed=42） | 18.88 | 24.78 | 0.512 |
| Decoder-only Transformer（seed=42） | 18.76 | 24.34 | 0.529 |

任务 2/3 首轮 Transformer 验证 RMSE 最低；任务 4 的三种子结果见下文，最终测试尚未进行。

## 任务 4 小规模调参与多种子实验

任务 4 已运行完成，结果在 `results/stage4/`，中文解释见 [docs/stage4_results.md](docs/stage4_results.md)。每个模型固定比较三个配置，以种子 42 的验证 RMSE 选择后，再追加种子 43、44，共十次训练；测试集未评价。

若只想查看结果，打开 `results/stage4/model_comparison.csv` 或 `summary.json` 即可，无需重新训练。完整重跑使用新的输出目录：

```text
python run_stage4.py --data "data/chengdu_20210701-20260701_sorted.xlsx" --output "results/stage4_my_run"
```

实验入口会在训练前记录计划、数据和源代码指纹，保存所有候选结果。已完成实验可以用以下命令重新加载并检查，或在中断后继续尚未开始的实验：

```text
python run_stage4.py --data "data/chengdu_20210701-20260701_sorted.xlsx" --output "results/stage4" --resume
```

`--resume` 要求计划、数据和相关源代码与原实验一致；若某一次训练已开始但未完整保存，程序会拒绝覆盖该非空目录，改用新的输出目录完整重跑即可。

当前选定配置为 Transformer 原始 32 维、学习率 0.001；LSTM 32 隐藏单元、学习率 0.0003。LSTM 的任务 3 默认命令继续保持学习率 0.001，以便复现旧结果；单独训练选定 LSTM 时须显式增加 `--lr 0.0003`。不要将任务 3 的旧权重当成任务 4 的选定权重。

## 任务 5 验证误差分析

任务 5 已完成。中文结论及汇报素材见 [docs/stage5_error_analysis.md](docs/stage5_error_analysis.md)，图片和表格保存在 `results/stage5/`。分析只使用验证结果，不重新训练或选模型，也不评价测试集。

重点看 `change_error_comparison.png`（普通日与突变日）、`largest_error_case.png`（最大误差案例）和 `training_curves.png`（早停依据）。`validation_comparison.png` 展示整个验证时段。逐日图固定采用 seed=42；分组神经网络指标汇总三个种子。

如果需要复现分析，使用新的输出目录：

```text
python analyze_validation.py --data "data/chengdu_20210701-20260701_sorted.xlsx" --experiments "results/stage4" --baselines "results/baselines" --output "results/stage5_my_run"
```

大幅变化阈值约为 44.8 个 AQI 点，来自训练期日变化的第 90 百分位。目标日的实际变化仅用于事后分组，不能作为预测输入。不能把整体 RMSE 较低表述为在所有日期或场景下都更好。

## WEEK 13 提交材料

- [英文一页报告 Word 版本](reports/Chengdu_PM25_Week13_Report.docx)：已用 WPS 核对为一页，包含当前模型、流程、验证进展、问题和下一步计划。
- [复现与提交说明](docs/reproduction.md)：完整运行顺序、各代码文件职责及提交范围。
- [任务 4 结果](docs/stage4_results.md) 与 [任务 5 分析](docs/stage5_error_analysis.md)：完整验证证据和限制。

本地 reports/PM25_Week13_Code.zip 是代码与文档备份，不包含数据、权重或缓存。实际实验输出仍保留在本地 results 中。

## 还未完成的工作

- PPT：按当前安排留待后续制作。
- 最终测试：在最终评价方案完全固定后进行。
- 如加入其他污染物，重新检查特征缺失，并使用相同评价日期比较。

以上为后续计划。任务 2、3 的表格为首次实验，任务 4 的选定配置与多种子汇总见上文入口及 docs/stage4_results.md；所有结果仍为验证集结果。

## 来源与数据保存

来源：https://aqicn.org/historical/cn/#city:chengdu 。感谢 WAQI 和原始环境监测机构；数据未经 WAQI 充分验证。详细含义及限制见 `data/README.md`。

本地工作目录可以包含 Excel 和运行结果；公开 GitHub 仓库应排除这些文件，`.gitignore` 已配置相应规则。每位成员使用自己的下载文件复现，不向公开仓库提交数据。Python 依赖范围见 `requirements.txt`；当前实际运行版本记录在每次输出的 `summary.json` 中。
