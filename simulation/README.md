# 第一次用专家脚本采集

专家脚本根据场景和指定部位规划右臂动作，再用 Isaac Sim 实际执行并录制。第一次可以用同一只棕色带柄杯录两条，一条夹杯身，一条夹杯柄。

## 专家脚本在哪里

| 找什么 | 文件 |
| --- | --- |
| **直接录一对示教** | [record_expert_demo.py](record_expert_demo.py) |
| 棕杯杯身和杯柄的**专家规划** | [step1_mug_pairs.py](isaac_snapshot/step1_mug_pairs.py) |
| 实际**录制、回放和验收** | [step3a_verify.py](isaac_snapshot/step3a_verify.py) |
| 生成并调度**整批新示教** | [step3b_collect.py](isaac_snapshot/step3b_collect.py) |

日常先用第一行。它调用原项目的录制与回放流程，直接打印两条数据的文件路径。准备更多物体时，再查看 [任务清单](../docs/datasets.md) 和 [批量采集操作](operations.md#批量运行与恢复)。

## 一条命令采两条轨迹

在已配置 Isaac Sim 的实验服务器上，打开终端并进入从 GitHub 下载的仓库根目录。下例使用原实验服务器的 Python 环境和仿真工程目录。第一次可在仿真机器空闲时运行。

```bash
/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python simulation/record_expert_demo.py --project-dir /data/sundawei/g1d_grasp_worker
```

在另一台已配置的服务器上，把命令里的 Python 和 `--project-dir` 改成本机路径。`--project-dir` 指向装好资产、相机标定和场景计划的**仿真工程**。仅下载 GitHub 仓库时，可以先观看已有演示。 [仿真安装与交接文件](operations.md#运行环境与交接文件) · [模型演示 MP4](../docs/demos/model-io.mp4)

想先检查路径，在命令末尾加 `--check`。它会检查一对现成专家计划和所需的工程文件。去掉 `--check` 才会启动仿真录制。

## 结果在哪里

- 终端显示 `PASS: 2/2` 表示杯身和杯柄都录制成功，重放后能抬起杯子，且夹到了指定部位。
- 终端在 `mug_body` 和 `mug_handle` 后打印两条 `.h5` 的完整路径。文件中有头部、左右腕画面和实际动作。
- `Result:` 后是验收 JSON 的完整路径。失败时先看这里，再按终端给的诊断路径找对应场景。

示例借用一对已验真的开发场景。生成的数据用于熟悉操作，训练数据仍按自己的采集清单和分区管理。

[返回仓库首页](../README.md) · [专家与仿真进阶操作](operations.md) · [全部脚本索引](isaac_snapshot/INDEX.md)
