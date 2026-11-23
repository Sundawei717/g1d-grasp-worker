# 数据管线与换模型指南

[返回仓库首页](../README.md) · [数据清单](datasets.md) · [V3-A 模型](model-v3a.md) · [采集入口](../simulation/README.md)

仓库的可安装包可直接读取和核验已验收的示教，向任意模型提供 NumPy 样本。仿真、Isaac Sim、专用权重和检测器都在这个接口之外。真实样本需要原项目的 HDF5 文件及其清单。

## 导入既有清单

从本仓库根目录运行。将示例中的 `/path/to/original` 换成旧工程根目录。它应同时包含 `runs` 和清单引用的轨迹。

```bash
python -m pip install -e ".[test]"
mkdir -p data

g1d-data import-manifest /path/to/original/runs/tier1_v3_01/training_split.json \
  data/tier1.json --source-root /path/to/original

g1d-data audit data/tier1.json --data-root /path/to/original \
  --protected benchmarks/development_identity.json \
  --protected benchmarks/l1_l4_identity.json

g1d-data inspect data/tier1.json --data-root /path/to/original \
  --split train --index 0 --history 2 --horizon 8
```

PowerShell 可逐条单行执行，目录可用 `New-Item -ItemType Directory -Force data` 创建。导入保留原始分区，同时将清单中的绝对 HDF5 路径转换为相对路径。移动数据时将 `--data-root` 指向具有同一相对目录结构的新根。Tier-2 合库完成后，把导入源换成 `runs/step3b_01/training_split.json`。

新样本尚未分区时，用下列命令划分可移植清单。

```bash
g1d-data split data/collection.json data/new_split.json --validation-fraction 0.20 --seed 33889
```

切分将相关的配对、物理布局和场景族整组分配，保留已有归属。若固定分区本身冲突，工具会报错。实际验证比例受组大小影响。正式 Tier-2 使用采集调度器已写定的分区。

## HDF5 时间轴与验收

- 文件使用 `g1d_v3_step_aligned`，一条轨迹有 `T+1` 条观测、`T` 条动作。`obs/control_step_id[t]` 与 `action/control_step_id[t]` 都为 `t`。动作是从观测 `t` 出发实际下发的右臂绝对角度与夹爪开口，末条观测保存最终反馈。
- 头部路径为 `obs/rgb_left`，左右腕为 `obs/left_wrist/rgb` 和 `obs/right_wrist/rgb`。三路都是 `uint8` 的 `224×224×3` RGB、gzip 无损图像，并按 20 Hz 对齐。状态在 `obs/qpos`、`obs/gripper_width`、`obs/eef_pose`。末端姿态原档为 XYZ 和 `xyzw`。
- `meta/initial_prompt_json` 保存首帧 BBox 和抓取点，均归一化到 0–1。`phase` 顺序为 `approach → close → lift`。`train_action_valid` 覆盖全部有效接近和闭爪动作，抬升只供验收回放使用。
- `audit` 校验 SHA-256、观测和动作对齐、状态与提示有效性、600 Hz 原生指令的绑定，以及与留出标识的重合。部分旧条目无基础网格哈希，覆盖数可查看审计的 `training_rows_with_mesh_hash`。旧档重编号时按清单身份索引，核对 case、物体、部位、步数与哈希。审计的 `archive_id_aliases` 给出重编号计数。

## 接到自己的模型

```python
from g1d_grasp.data import WindowDataset, load_manifest

manifest = load_manifest("data/tier1.json")
train = WindowDataset(manifest, "/path/to/original", split="train", history=2, horizon=8)
validation = WindowDataset(manifest, "/path/to/original", split="validation", history=2, horizon=8)
example = train[0]
print(example["rgb"]["head"].shape, example["actions"].shape)
```

| 样本字段 | 含义 |
| --- | --- |
| `rgb` | 当前步的 `head`、`left_wrist`、`right_wrist`，各 `[224,224,3]` |
| `prompt.bbox`、`prompt.point` | 首帧归一化 `[x1,y1,x2,y2]` 和 `[u,v]`。另有资产查询字符串 `target_query` |
| `history` | 相邻的 H 帧控制步、7 关节位置（rad）、开口（m）、末端 XYZ 与 xyzw。首帧向前补齐 |
| `actions` | 从当前步开始的 `[K,8]` 动作，7 关节目标（rad）及 1 夹爪开口目标（m） |
| `action_mask`、`close_mask` | `[K]` 可监督位置和其中的闭爪位置。未来跨度进入试提或越界时填零并屏蔽 |
| `episode_id`、`observation_step`、`timestamp_s` | 轨迹身份和观测时间索引 |

可以调整 `history` 与 `horizon`。替换模型时从这些原始单位开始定义自己的编码与归一化，统计量只从训练分区拟合。如果继续接同一执行器，提供可还原成绝对关节目标和开口的动作，沿用清单关节顺序、20 Hz 时间轴与掩码。 [模型替换示例](../examples/train_custom.py)

```bash
python -m pip install -e ".[v3a]"
python examples/train_custom.py data/tier1.json --data-root /path/to/original --max-batches 4
```

这个最小示例跑若干批训练，验证从取样到反向更新的接口。正式 V3-A 训练及闭环由仿真目录的旧工程脚本衔接，模型换代后需要在仿真端实现对应的在线状态编码、模型加载和动作输出适配，再用控制回放对齐验证。

## 源码导航

| 文件 | 职责 |
| --- | --- |
| [episode.py](../src/g1d_grasp/data/episode.py) | 读取 HDF5 字段、检查时间轴、形状、掩码和哈希 |
| [manifest.py](../src/g1d_grasp/data/manifest.py) | 导入路径、检查数据身份、整组切分、对照保护清单 |
| [windows.py](../src/g1d_grasp/data/windows.py) | 将真实轨迹索引成模型中立的训练窗口 |
| [cli.py](../src/g1d_grasp/data/cli.py) | 清单导入、审计、切分、抽样查看的命令行 |
| [test_data_pipeline.py](../tests/test_data_pipeline.py) | 用小型合成档案覆盖时序、掩码、切分和留出检查 |
