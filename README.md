# G1D 三视角抓取

这个仓库把 G1D 桌面抓取项目的**数据管线、模型和仿真运行**放在三个独立的入口里。更换模型时可以继续用同一批已验收的 HDF5、同一分区与同一评测清单。

## 目录

| 路径 | 用途 |
| --- | --- |
| [src/g1d_grasp/data](src/g1d_grasp/data/) | 可移植清单、校验、防泄漏切分、NumPy 训练窗口 |
| [src/g1d_grasp/models](src/g1d_grasp/models/) | 可选 V3-A 模型、BBox 与抓取点编码、检查点推理 |
| [examples](examples/) | 接入新模型的最小训练示例、V3-A 单样本推理 |
| [simulation](simulation/) | Step 3-B 采集、训练、闭环和恢复脚本的原运行快照 |
| [benchmarks](benchmarks/) | 开发集及 L1–L4 冻结场景的标识清单 |
| [docs](docs/) | 实验口径、指标与模型输入输出演示 |

仓库收录源码、说明与小型演示文件。轨迹、模型权重、仿真资产、设备校准和运行日志从原数据盘单独取得。

## 快速开始

Python 3.10 或更新版本。数据安装仅需要 NumPy 和 h5py。

```bash
python -m pip install -e ".[test]"
python -m pytest
```

### 1. 将采集清单转为相对路径

```bash
g1d-data import-manifest /path/to/original/runs/tier1_v3_01/training_split.json \
  data/train.json --source-root /path/to/original

g1d-data audit data/train.json --data-root /path/to/original \
  --protected benchmarks/development_identity.json \
  --protected benchmarks/l1_l4_identity.json
```

`--source-root` 是源 HDF5 路径共同所在的目录。导入只转换清单，源 HDF5 与清单原有分区保持不变。迁移数据后在 `--data-root` 指定保存同一相对目录结构的新位置。Windows 可以将上述命令写成单行执行。

`audit` 校验 HDF5 SHA-256、20 Hz 对齐、三路无损图像、接近和闭爪标签、600 Hz 原生指令衔接、数据身份及已记录的分区防泄漏字段。若某些旧条目缺少网格哈希，结果中的 `training_rows_with_mesh_hash` 会显示校验覆盖范围。旧样本重新编号后，以正式清单的轨迹和场景 ID 为准，由 case_id、物体、部位、步数和 HDF5 哈希绑定原档。archive_id_aliases 统计重新编号的条数。

新采集清单中待分配的组可用以下命令处理。已经划分的组沿用原归属，剩余组按固定随机种子和组粒度分配，实际比例受组大小影响。

```bash
g1d-data split data/collection.json data/train.json --validation-fraction 0.20 --seed 33889
g1d-data inspect data/train.json --data-root /path/to/original --split train --index 0
```

Step 3-B 完成后，直接导入其输出的 `runs/step3b_01/training_split.json`，保留 Tier-2 正式实验分区。

### 2. 给新模型加载数据

```python
from g1d_grasp.data import WindowDataset, load_manifest

manifest = load_manifest("data/train.json")
train = WindowDataset(manifest, "/path/to/original", split="train", horizon=8, history=2)
sample = train[0]
print(sample["rgb"]["head"].shape)   # (224, 224, 3)
print(sample["actions"].shape)      # (8, 8)
print(sample["action_mask"])         # True 对应可监督的目标动作
```

| 字段 | 形状或含义 |
| --- | --- |
| `rgb` | 当前观测的 `head`、`left_wrist`、`right_wrist`，各 `224×224×3`、uint8 |
| `prompt.bbox`、`prompt.point` | 首帧归一化 `[x1,y1,x2,y2]` 和 `[u,v]` |
| `history` | 最近 H 帧的关节（rad）、夹爪开口（m）、末端位姿（XYZ 与 xyzw）和控制步 |
| `actions` | `K×8` 未来动作，7 个绝对关节角目标（rad）与 1 个夹爪开口目标（m） |
| `action_mask`、`close_mask` | 有效动作及其中的闭爪动作，形状均为 `K` |
| `episode_id`、`observation_step`、`timestamp_s` | 源轨迹身份、当前控制步和时间戳 |

前缀历史不足时用首帧补齐。未来窗口在最后一个闭爪动作结束，后续填零并由掩码排除。训练和验证分别创建数据集。归一化统计量在训练集拟合后可用于两者。

模型只需根据样本生成 `[B,K,8]` 目标或提供自己的动作适配层。损失计算应使用 `action_mask` 排除填充，可按 `close_mask` 调节闭爪权重。[最小训练示例](examples/train_custom.py) 演示了前向、掩码与反向更新。

```bash
python -m pip install -e ".[v3a]"
python examples/train_custom.py data/train.json --data-root /path/to/original --max-batches 4
```

### 3. 用原 V3-A 检查点做基线对照

V3-A 使用头部 RGB、BBox 掩码、抓取点热图和双腕 RGB。参考适配器保持原检查点的三路编码、两帧 17 维机器人状态及动作反归一化。

```bash
python examples/predict_v3a.py data/train.json /path/to/last.pt \
  --data-root /path/to/original --split validation --index 0 --device cpu
```

命令输出一次前向的 8 步关节目标与夹爪开口。正式闭环的时序融合、统一试提和 Part Hit 判定位于仿真代码中。

## 实验入口

当前基线和冻结清单的口径见 [实验说明](docs/experiments.md)。原始控制器入口及外部 Isaac Sim 依赖见 [仿真运行说明](simulation/README.md)。

- [模型输入和输出 HTML](docs/demos/model-io.html) 可下载后离线打开。绿色 BBox 固定，选择杯身或杯柄查看真实预测。
- [模型输入和输出 MP4](docs/demos/model-io.mp4) 演示同一观测上的换点和 8 步动作对照。
