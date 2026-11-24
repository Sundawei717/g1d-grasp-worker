# 批量采集指南

批采是让专家反复更换物体、抓取部位、物体角度、机器人起姿和障碍位置。每个新场景先规划，再在 Isaac Sim 中录制并重放，验收通过才计入训练数据。想先体验一组，见 [单组采集入门](README.md)。

## 批采会生成哪些数据

当前任务保留第一批的 460 条，计划用 12 款资产再录 **1,140 条**。 [12 款资产与部位清单](../docs/datasets.md)

| 按什么变化 | 实际怎么生成 | 新增预算 |
| --- | --- | ---: |
| 换目标或换部位 | 同一只杯子换杯身／杯柄，同一只篮子换提手／侧沿，同一长盒换两端。白盒和罐子等也可以同时摆在桌上，每次只指定其中一个抓取 | 456 |
| 绕障 | 在杯、篮、长盒等目标前放入不同尺寸的罐、挡块或挡板。规划时先确认直达路径真被挡住，再找一条可行的绕行路径。对称可行时优先右臂外侧 | 400 |
| 换角度和起姿 | 从已验收位姿出发，改变目标在桌面上的位置与偏航角，并改变右臂初始关节角，再计算新的抓取路径。先前的杯、篮覆盖 −75° 至 +75°，长盒覆盖 −90° 至 +90° | 284 |

采集器先用 `schedule()` 分配资产和任务类型，再用 `build_group()` 为每组两条安排场景。每次试新角度会重算对应的物体位姿、抓取和抬升路径。放两个目标时，两次录制互换目标与同桌物体。规划失败或物理验收失败的候选会换布局重试。成功的示教保存为三路无损 HDF5，关联场景按组分区，开发集和冻结测试集保持独立。 [采集实现](isaac_snapshot/step3b_collect.py)

启用动态障碍时，`runs/step3b_01/fixture_physics.json` 的 `dynamic_from_batch` 指定从哪一批开始。挡路物和同桌干扰物受重力和接触力作用。录制与重放使用该批次的同一设置，HDF5 场景元数据和验收回执用 `fixture_physics` 标记。

自适应夹爪通过 `runs/step3b_01/gripper_gate.json` 的 `adaptive_from_batch` 启用。批次记录 `gripper_mode`，原始动作回放也按同一版本运行。新门控从实际开口判断是否继续夹紧，常规夹紧余量为 2 mm，接近机械闭限时降至剩余行程的一半。两指垫持续建力、开口稳定和右臂停稳后才试提。

当前工程从第 23 批启用以上两项设置；已经落盘的批次继续使用各自记录的版本。配置文件格式、默认行为和 28 场复测结论见 [夹爪与障碍版本说明](gripper-control.md)。

## 启动或管理批采

下面的命令在 **实际运行总控的 Linux 主机**、已配置的仿真工程目录执行。2026-11-24 的部署是在 114 上计算，通过挂载将结果写入 115 的同一路径。查看进程、暂停和续跑应在 114 操作；115 用于保存数据和查看文件。GitHub 仓库保存代码，资产、相机标定和实际任务清单由原运行环境提供。 [环境交接清单](operations.md#运行环境与交接文件)

先查任务状态。如果已显示 `RUNNING`，从这里看进度即可。

```bash
cd /data/sundawei/g1d_grasp_worker
SIM_PYTHON=/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python
"$SIM_PYTHON" step3b_control.py status
```

**首次启动整套流程**，或已搭建新的兼容运行环境时，先预检，然后启动。当前总控会在采满合库后**自动进入训练和评测**。

```bash
"$SIM_PYTHON" step3b_control.py preflight
"$SIM_PYTHON" step3b_control.py start
```

**看日志**。下面的命令持续跟踪输出，按 `Ctrl-C` 退出日志查看。

```bash
tail -f runs/step3b_01.log
```

**暂停**。采集会让已开始的分片结束并落盘。

```bash
"$SIM_PYTHON" step3b_control.py pause
```

**续跑**。用 `status` 确认已暂停或失败退出后再恢复。已验收的组会自动跳过。

```bash
"$SIM_PYTHON" step3b_control.py resume
```

## 看进度与完成标志

```bash
"$SIM_PYTHON" step3b_control.py status
python3 background_status.py step3b_01_stage1 step3b_01
```

| 想知道什么 | 看哪里 |
| --- | --- |
| 采到多少条、各物体采了多少 | `status` 或 `runs/step3b_01/stage1_progress.json`，其中新增数的分母是 1,140 |
| 一批为何少了几组 | `runs/step3b_01/collect/batches/` 中对应批次的 `failures` 和诊断路径 |
| 哪些组已经验收 | `runs/step3b_01/collect/accepted/`。组回执给出 HDF5 路径、哈希和回放证明 |
| 采集全部结束 | `runs/step3b_01_stage1.complete` 的 `outcome` 为 `PASS` |
| 最终用什么清单训练 | `runs/step3b_01/training_split.json`。旁边的 `data_report.md` 汇总各资产和分区 |
| 整套采集、训练和评测结束 | `runs/step3b_01.complete` 的 `outcome` 为 `PASS` |

## 想扩展自己的任务清单

1. **只想看看另一种物体或角度**时，在 [单组入门页](README.md) 换预设，先核对录像和夹取部位。
2. **想新增指定角度、起姿或障碍**时，复制一份同资产的场景定义，调整目标偏航与位置、`home_q` 或 `fixtures`。角度写入物体的 `target_pose_xyz_xyzw`，`yaw_deg` 用于记录。用 [规划器](isaac_snapshot/step3a_planner.py) 的 `placement()`、`solve()` 生成新的位姿和可执行计划。带挡路物时参照 [批采代码](isaac_snapshot/step3b_collect.py) 的 `with_obstacle()` 检查是否需要绕行。沿用示例的录制、重放和 Part Hit 验收接口，先验证一对或一小批。
3. **想换整批配额或引进新网格**时，先做资产碰撞与部位标注验收，然后在独立的新实验运行目录调整 `step3b_collect.py` 里的 `ASSET_ORDER`、`schedule()` 和 `MODE_TARGETS`，同时更新合库预算和预检。当前运行的计划已落盘，续跑时会核对原配额。新任务使用新的场景、轨迹和配对 ID。

[总控与资产进阶说明](operations.md) · [训练数据格式](../docs/data-pipeline.md)
