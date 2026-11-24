# 夹爪控制、动态障碍与复测

[专家采集入门](README.md) · [批量采集](batch-collection.md) · [完整复测报告](../docs/reports/2026-11-24-a40-gripper.md)

## 这次修复了什么

策略预测的夹爪目标宽度会有小幅抖动。旧逻辑只有在“本次目标不大于上次目标”时才保留夹紧预载，导致目标从 41.0 mm 轻微增到 41.2 mm 时，实际下发却可能变成 46.0 mm。新逻辑判断目标是否仍比实测开口窄，并限制下发指令每个控制步最多变化 4 mm。请求仍是夹紧时，不会因为这点抖动直接退回较宽开口。

本次还加入不依赖物体/部位类别的试提门控，以及受重力和碰撞影响的动态障碍采集。模型结构、权重和数据加载接口没有变化。

## 两种夹爪版本怎么选

| 项目 | `legacy`：重现历史对照 | `adaptive`：本轮修复 |
| --- | --- | --- |
| 保留夹紧预载 | 检查请求是否不大于上一指令，保留旧行为 | 检查请求相对实测开口的夹紧意图，限制指令变化速度 |
| 持续闭爪判定 | 下发开口 ≤ 71 mm 持续 0.25 s | 下发开口相对实测开口保有夹紧意图，持续 0.25 s |
| 夹紧余量 | ≥ 6 mm | ≥ `min(2 mm, 剩余可闭合行程 / 2)` |
| 双指垫建力 | 每侧 ≥ 5 N | 每侧 ≥ 5 N |
| 空夹到机械闭限 | 可触发试提，通常随后失败 | 单凭空夹触底不触发试提 |

两种版本均保留 0.20 s 手臂稳定窗口、开口变化 <0.5 mm 持续 0.15 s、8 cm 试提、0.55 s 悬停和 15 秒总仿真上限。试提锁定最后下发的夹爪目标指令。

**入口默认行为需要区分：**直接创建 `LiftGateConfig()` 得到 `adaptive`；共用分片接口 `step2_dev.simulate()` 默认仍为 `legacy`；旧清单没有 `gripper_mode` 时也按 `legacy` 读取。这样原数据回放和历史对照仍能使用原控制口径。需要修复版本时，显式传入 `gripper_mode="adaptive"`，或在清单的 `unified_lift_gate` 中写入该字段。仅更新 Python 文件不会把所有旧计划切换成新版本。

2026-11-24 审查补丁已让 `step3b_eval.py` 正式评测入口显式选择 `adaptive`，正式运行时会重新测量同版 A40 基线；结果按执行合同缓存。补丁已同步到 GitHub，并在本地与服务器暂存目录通过验证，尚未切换常驻生产总控。详见 [审查报告](../docs/reports/2026-11-24-pipeline-audit.md)。

## 批采如何启用

当前运行目录使用两个小配置文件：

| 文件 | 本次内容 | 作用 |
| --- | --- | --- |
| `runs/step3b_01/fixture_physics.json` | `{"dynamic_from_batch": 23}` | 从新建的第 23 批起，让挡路物和同桌干扰物成为动态刚体 |
| `runs/step3b_01/gripper_gate.json` | `{"adaptive_from_batch": 23}` | 从新建的第 23 批起，用修复后的夹爪版本 |

以上是当前部署值，新实验应选择自己的起始批号。配置不存在时，新批次仍选 `kinematic` / `legacy`。每批生成时把版本固化在批次清单里，录制、原始指令回放与验收回执要一致。修改起始批号不会重写已经落盘的批次；不要为续跑去改写旧回执或旧 HDF5。

动态障碍在当前代码中用于专家录制和原生回放。初态要经过物理稳定检查，试提预检读取障碍实时位姿。旧的竖放障碍解锁后可能自行翻倒，这类初态会被拒收。此次 A40 开发集复测仍保持原来的固定障碍，以单独比较夹爪改动。

## 28 场复测如何运行

下列是 **2026-11-24 固定实验**的入口，依赖原 Linux 仿真工程、已有 A40 权重、开发场景计划及历史对照结果。只克隆仓库不足以启动仿真，环境交接见 [进阶操作](operations.md#运行环境与交接文件)。脚本按原工程的顶层布局部署。

```bash
cd /data/sundawei/g1d_grasp_worker
SIM_PYTHON=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
"$SIM_PYTHON" eval_gripper_repair.py --preflight
"$SIM_PYTHON" eval_gripper_repair.py
```

脚本校验 A40 权重、开发清单、每场计划和关键代码哈希；按资源余量派发仿真，并复用已完成分片的回执以续跑。它只做评测，不启动训练。原运行已经完成时会直接返回，不重复测同一份结果。修改实验条件应使用独立运行名称和目录。

```bash
"$SIM_PYTHON" eval_gripper_repair.py --status
cat runs/step3b_01_gripperfix_a40_dev28_20261124.complete
```

`progress.json` 给出已完成场数和当前状态。只有全部 28 场结束且输入配对检查通过，才写入 `.complete`；`PASS` 表示评测流程完成，不表示 28 场全抓取成功。

评测完成后，可用本次固定实验的辅助脚本核对物理记录并生成中文报告：

```bash
"$SIM_PYTHON" audit_gripper_repair.py
"$SIM_PYTHON" report_gripper_repair.py
```

结果均位于 `runs/step3b_01_gripperfix_a40_dev28_20261124/`：`result.json` 保存完整指标，`physical_audit.json` 保存指令跳变和试提审计，`report.md` 与 `comparison.csv` 提供阅读版。报告辅助脚本包含本次固定案例的解释，不应直接套用到另一批实验。

## 本轮结果与测试

同一个 A40 `last.pt` 的抓起数为 **19/28 → 19/28**，抓起且部位正确为 **16/28 → 15/28**；下发开口单步增加超过 4.01 mm 的事件为 **19 → 0**。控制跳变已消除，整体成功率没有提高。完整分层和逐场结果见 [复测报告](../docs/reports/2026-11-24-a40-gripper.md)。

[夹爪回归测试](../tests/test_unified_grasp.py) 覆盖 41.0→41.2 mm 抖动、短暂失力、主动张爪、薄件/宽件边界、空夹和三步稳定要求。在仓库根目录按首页配置测试依赖后运行 `python -m pytest`。本轮本地共 15 项通过，其中 10 项为夹爪参数化测试。

| 文件 | 用途 |
| --- | --- |
| [unified_grasp.py](isaac_snapshot/unified_grasp.py) | 指令保护、夹紧判断和统一试提 |
| [lab_school_probe.py](isaac_snapshot/lab_school_probe.py) | 动态刚体、版本清单加载和实时抬升预检 |
| [step2_dev.py](isaac_snapshot/step2_dev.py) | 带版本校验的分片执行与续跑 |
| [step3b_collect.py](isaac_snapshot/step3b_collect.py) | 固化批次版本，统一录制、回放与入库 |
| [step3b_control.py](isaac_snapshot/step3b_control.py) | 将执行器加入启动前代码哈希检查 |
| [eval_gripper_repair.py](isaac_snapshot/eval_gripper_repair.py) | A40 固定 28 场复测与结果配对 |
| [audit_gripper_repair.py](isaac_snapshot/audit_gripper_repair.py) | 下发开口和物理失握审计 |
| [report_gripper_repair.py](isaac_snapshot/report_gripper_repair.py) | 本次复测的中文报告与 CSV |
