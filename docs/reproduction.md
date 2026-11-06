# WEEK 13 复现与提交说明

本阶段包含两个简单 baseline、Decoder-only Transformer、LSTM、固定小规模调参与三种子验证，以及事后验证误差分析。报告位于 reports/Chengdu_PM25_Week13_Report.docx。PPT 按当前安排留待后续，测试集尚未评价。

## 新电脑复现

按 README 创建 Python 3.12 的 pm25 环境，安装依赖。在 CPU 环境可先执行：

```text
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

从来源下载并自行保留对应 Excel，放到 data/chengdu_20210701-20260701_sorted.xlsx。研究范围为 2021-07-01 至 2026-07-01，列为 date 和 pm25，单位是 PM2.5 分项 AQI。源文件未随代码分发，数据下载说明见 data/README.md。

在项目根目录运行：

```text
python -m unittest discover -s tests -v
python pipeline.py --data "data/chengdu_20210701-20260701_sorted.xlsx"
python run_stage4.py --data "data/chengdu_20210701-20260701_sorted.xlsx" --output "results/stage4"
python analyze_validation.py --data "data/chengdu_20210701-20260701_sorted.xlsx" --experiments "results/stage4" --baselines "results/baselines" --output "results/stage5"
```

第一步应有十六项检查通过。新电脑首次复现时结果目录尚不存在；若已经运行过，请使用 README 中新的输出目录，或在相同源代码和数据下使用任务 4 的 --resume，避免混淆历史实验。若改变 baseline 输出目录，分析时也须通过 --baselines 指向对应目录。

同一数据的预期样本数为训练 1,261、验证 265、测试 256。任务 4 的三种子验证均值为 Transformer MAE 18.66、RMSE 24.31、R² 0.530；LSTM MAE 18.89、RMSE 24.74、R² 0.514。操作系统、数学库和 PyTorch 版本不同可能产生数值差异，应保留实际环境版本及全部实验记录。

## 文件职责

| 文件 | 职责 |
|---|---|
| pipeline.py | 日期检查、连续窗口、时间划分、简单 baseline |
| transformer_model.py / lstm_model.py | 模型结构及保存权重的加载 |
| sequence_training.py | 两种神经网络共用的数据准备、训练及验证协议 |
| train_transformer.py / train_lstm.py | 单配置训练入口 |
| predict_transformer.py / predict_lstm.py | 从连续七天 AQI 预测次日 |
| run_stage4.py | 固定候选调参、三种子复核及指标汇总 |
| analyze_validation.py | 验证误差分组、案例及训练曲线分析 |
| tests/ | 十六项检查 |

## 提交范围

公开仓库包含代码、依赖说明、测试、数据下载说明、阶段文档及一页报告。原始 Excel、日历或逐日预测 CSV、模型权重、结果图片、运行缓存以及本地代码压缩包不提交到公开仓库。结果均可通过上述入口在本地重新生成。

本地的 reports/PM25_Week13_Code.zip 是用于备份或向组员交付的代码与文档包，其中不包含原始数据、权重、results 或 Python 缓存。它不能替代课程要求的代码仓库链接。提交前应确认 GitHub main 中包含本阶段新实现。

最终测试未提供运行入口，留待最终方案固定后再推进。PPT 不在本次交付范围中。
