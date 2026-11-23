# G1D 三视角桌面抓取

这个项目训练 G1D 机器人的右臂和双指夹爪，按首帧指定的目标和部位完成接近与夹持。同一只带柄杯子可以分别指定杯身或杯柄，桌上有挡路物时还要绕行。策略根据头部相机、双腕相机和机器人状态预测动作，统一执行器负责确认夹稳、试提与判分。当前实验在 Isaac Sim 中完成。

[下载带 BBox 的模型输入输出 HTML](docs/demos/model-io.html) · [下载同款 MP4 演示](docs/demos/model-io.mp4)

## 想采数据，从这里开始

- **先录一组**。用 `simulation/record_expert_demo.py` 选择杯子、篮子、长盒、白盒与罐子等预设。也能选大偏航和绕障。 [专家采集入门](simulation/README.md)
- **批量录制**。按资产和任务配额生成新场景，验收后自动合库。 [批量采集指南](simulation/batch-collection.md)

## V3-A 和 V3-B 是什么

它们是同一个抓取模型的两种输入配置。**两者的头部都接收目标 BBox 和抓取点**，左腕都接收 RGB。V3-A 的右腕只看彩色图，V3-B 的右腕还多看一张检测器生成的目标框掩码。项目已选用 V3-A 做后续扩采和训练。 [对照说明与结果](docs/experiments.md#v3-a-和-v3-b-是什么)

## 当前进度

- 第一批数据 **Tier-1 已验收 460 条**，覆盖带 C 形把手的棕杯、带提手小篮、长盒、白盒和百事罐。
- 已选定 **V3-A** 作为扩量路线。左右腕只用 RGB，头部仍提供目标 BBox 和抓取点。用 460 条数据训练 40 轮的版本，在 28 个独立开发场景上 **19 次抓起，16 次抓起且命中指定部位**。
- 正在做第二批 **Tier-2 扩采**，目标是沿用 460 条并净增 1,140 条，总计 1,600 条。随后训练 80 轮，先在 28 个开发场景选权重，再与现有基线在冻结的 80 个任务上对照。各阶段状态、日期和进度文件见 [项目方案与进展](docs/experiments.md)。

## 从哪里开始

| 我想做什么 | 从这里读 |
| --- | --- |
| 了解项目方案、阶段进度、指标与实验结果 | [项目方案与进展](docs/experiments.md) |
| 了解 V3-A 结构、张量、BBox 与动作输出，或跑离线推理 | [V3-A 模型说明](docs/model-v3a.md) |
| 查有哪些物体、抓取部位、障碍与已验收数据 | [数据清单](docs/datasets.md) |
| 复用 HDF5、划分数据、接入自己的模型 | [数据管线与换模型指南](docs/data-pipeline.md) |
| 第一次用专家脚本录制 | [专家采集入门](simulation/README.md) |
| 批量采集、看进度、暂停或继续 | [批量采集指南](simulation/batch-collection.md) |
| 配置仿真或查底层脚本 | [仿真进阶操作](simulation/operations.md) |

## 本地快速检查

Python 3.10 或更新版本。数据模块使用 NumPy 和 h5py。以下检查使用合成 HDF5，无需仿真器或项目权重。

```bash
python -m pip install -e ".[test]"
python -m pytest
g1d-data --help
```

如果拿到项目真实数据，先按 [数据管线指南](docs/data-pipeline.md) 导入清单。用 [train_custom.py](examples/train_custom.py) 验证新模型能取数据、计算带掩码的损失并更新参数。用 [predict_v3a.py](examples/predict_v3a.py) 读取已有项目检查点并打印一次前向的 8 步动作。

仓库包含源码、基准标识和小型演示。真实 HDF5、模型权重、资产、相机标定及完整场景清单由项目数据盘单独交接。重新采集和闭环评测还需要 [原始 Linux 仿真环境及配套文件](simulation/operations.md#运行环境与交接文件)。
