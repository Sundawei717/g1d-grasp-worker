# 仿真进阶操作

[专家采集入门](README.md) · [仓库首页](../README.md) · [可采集的物体和任务](../docs/datasets.md) · [方案与进展](../docs/experiments.md) · [HDF5 与换模型](../docs/data-pipeline.md)

这里的专家是基于几何和物理反馈的规划器。它根据资产、初始关节状态和指定部位计算一组可执行的腕部位姿与关节路径，再在 Isaac Sim 中录制。采新资产时从少量配对样本开始，验收后再加入调度清单。

## 运行环境与交接文件

[isaac_snapshot](isaac_snapshot/) 保存正式实验的 Linux 源码快照，脚本采用原有平铺导入方式。仅处理 HDF5 或换模型训练时，用 [可安装的数据包](../docs/data-pipeline.md) 即可。要重新采集或评测，需要项目维护者一并交接以下内容。

| 所需内容 | 使用位置 |
| --- | --- |
| 带相机支持的 Isaac Sim 6.0 / Isaac Lab，原 sim60 Python 环境 | `run_lab_school.sh`、`lab_school_probe.py`、训练入口。包含数值计算与 Pinocchio；当前主链路所需的 `projects.g1d_rsi` 桥接源码现已附在 [runtime](runtime/README.md) |
| G1D/Dex1 URDF 与机器人 USD、桌面和物体 USD、标定、部位三角面图谱 | `env_resolved.json` 和计划文件中各路径。抓取路径筛选和 Part Hit 都使用这些几何 |
| 已验机器实测与场景基线 | `runs/lab_school_03/result.json` 以及统一执行模板 `runs/three_view_step0_01/plans/expert_carton.json` |
| 训练数据与专家计划 | `runs/tier1_v3_01/` 下的真实 HDF5、原清单、专家计划与回执 |
| 扩采资产验收回执与完整评测清单 | `runs/step3a_01/tier2_ready_assets.json`、同目录的小样本计划/回执、`runs/step2_v3_01/development.json`、`runs/step3a_01/frozen.json` |
| 从原运行续跑时的产物 | `runs/step3b_01/`、同名前缀的日志与完成标志、A40 参考权重，以及冷启动用的 ResNet-18 权重 |

新机需还原目录布局或适配脚本中的路径。尤其检查 `step3b_common.py` 的解释器与运行根目录、`run_lab_school.sh` 的项目与安装根目录，以及 `env_resolved.json` 引用的校准和资产。最终冻结清单有 SHA-256 校验，迁移时需保留文件内容和清单所用的资产路径映射。仓库的身份清单用于查重，完整场景需用交接资料。配置、数据和权重放在项目数据盘。

以下运行命令从**装好上述依赖的原 Linux 仿真工程根目录**执行，脚本文件位于该目录顶层。先把 `SIM_PYTHON` 指向与此部署匹配的 sim60 解释器。确认相机、Vulkan GPU、显存和主机内存可用。

```bash
export SIM_PYTHON=/path/to/sim60/bin/python
bash -n run_lab_school.sh
"$SIM_PYTHON" step3b_control.py --help
nvidia-smi
free -h
```

## 专家采集的四步

1. **准备场景**。配置腕姿候选、目标位姿 `target_pose_xyz_xyzw`、右臂起姿 `home_q`、`seed`、`part` 及 `fixtures`。先用已验资产规则选择候选抓取姿态，解逆运动学，检查接近、绕障、夹爪开合与持物抬升通路。输出每场 `plan.json` 和供仿真加载的分片清单。新杯柄、篮提手先做凸分解和物理射线贯通检查，再标注原网格三角面归属。
2. **录制**。用 `three_view_expert_v3` 分片在物理仿真中执行示教。保存 20 Hz 三路画面、反馈与下发动作，以及用于精确复放的原生指令。阶段为接近、闭爪、试提。首帧框从目标投影获得，点对应指定抓取部位。
3. **重放原始动作**。用 `three_view_unified_replay` 分片和录制回执中的 HDF5 路径及 SHA-256 重新复位。统一执行器检验时间戳、下发指令和试提触发。通过条件包括抓起、Part Hit、控制步数不超过 300、时间戳误差 ≤ 0.2 μs、原生指令误差 ≤ `1e-6`、触发相差 ≤ 0.10 s。
4. **按组验收入库**。杯身／杯柄等部位配对还要检查初态、首帧框相同，点确实不同。两条都过审后写入接收回执，才计入采集进度。保留诊断、场景签名、部位和配对 ID，合库时按组切分。

执行器按清单中的 `gripper_mode` 区分版本。`legacy` 保存历史基线口径：开口指令 ≤ 71 mm 持续 0.25 s，双指垫各 ≥ 5 N 且夹紧余量 ≥ 6 mm；无接触的空抓可走 ≤ 2 mm 机械闭限分支。`adaptive` 使用请求/下发开口相对实测开口判断夹紧意图，限制指令每步变化不超过 4 mm，并按剩余机械行程调整余量要求；空夹触底本身不能触发试提。旧清单未标版本时按 `legacy` 解释，新批采和新评测需显式选择 `adaptive`。[两种版本、默认行为与复测方法](gripper-control.md)

两种版本均要求右臂停稳：0.20 s 内末端最大位移 < 10 mm 或最大关节速度 < 0.10 rad/s；开口每步变化 < 0.5 mm 持续 0.15 s。触发后保持最后下发的夹爪目标，试提 8 cm、悬停 0.55 s，统一受 15 秒时限约束。

[统一试提控制](isaac_snapshot/unified_grasp.py) · [录制与回放校验](isaac_snapshot/step3a_verify.py)

## 手动运行已准备的分片

第一次录一组示例，使用入门页的一条命令。以下入口供已经生成分片配置和清单的人使用。`JOB_DIR` 是该项目 `runs/` 下的新目录。 [专家采集入门](README.md#录一组你想看的场景)

```bash
G1D_WORKER_CONFIG="$SHARD_CONFIG" bash run_lab_school.sh "$JOB_DIR" \
  --grouped-manifest "$SHARD_MANIFEST" \
  --grasp-after runs/lab_school_03/result.json --gpu 0
```

出错时检查单场目录的 `expert_diagnostic.json`。其中含 `policy_rollout`、门控状态、试提原因、接触与阶段信息。再读同目录的 `physics_trace.jsonl` 或对应分片日志。

## 批量运行与恢复

完整的任务配比和日常命令见 [批量采集指南](batch-collection.md)。

总控把扩采合库、训练、评测串行衔接。首次启动或代码和输入变更后先运行预检。它检查资产隔离、去重、分片和两目标模板，运行训练入口的小规模前向与分片数学检查，并检查评测绑定及报告生成。预检通过后在 tmux 脱离终端运行。

```bash
"$SIM_PYTHON" step3b_control.py preflight
"$SIM_PYTHON" step3b_control.py start
"$SIM_PYTHON" step3b_control.py status
tail -f runs/step3b_01.log
```

已有任务日常先用 `status` 查看。进入控制窗口用 `tmux -L g1d-step3b attach -t step3b_01`，安全脱离按 `Ctrl-b` 再按 `d`。暂停会等待正在运行的采集分片结束并落盘。总控退出后，用同一部署与工作目录恢复。

```bash
"$SIM_PYTHON" step3b_control.py pause
"$SIM_PYTHON" step3b_control.py status
"$SIM_PYTHON" step3b_control.py resume
```

| 文件（位于原工程运行目录） | 查看什么 |
| --- | --- |
| `runs/step3b_01/controller.json`、`runtime.json` | 当前阶段、进程状态、资源水位、租用 GPU 与 Kit 清理回执 |
| `runs/step3b_01/stage1_progress.json` | 每个资产和采集模式已验收多少条 |
| `runs/step3b_01/training_split.json`、`data_report.md` | 合库与分区结果 |
| `runs/step3b_01/stage2_progress.json`、`fit/loss_curves.csv` | 当前轮次与训练／验证损失。`fit/` 在同一工作目录下 |
| `runs/step3b_01/stage3_progress.json`、`evaluation/adaptive_gripper_v2/selection.json` | 正在评哪组、选中了哪个检查点。`evaluation/` 在同一工作目录下；旧部署的结果仍在 `evaluation/` 顶层 |
| `runs/step3b_01_stage{1,2,3}.complete`、`runs/step3b_01.complete` | 阶段和全流程完成回执。读取内容确认 `outcome` 为 `PASS` |
| `runs/step3b_01/final_report.md` | 完成冻结评测后生成的分层结果 |

以下命令显示各阶段完成标志和本机资源。

```bash
python3 background_status.py step3b_01_stage1 step3b_01_stage2 step3b_01_stage3 step3b_01
nvidia-smi
free -h
df -h /data /dev/shm
```

## 入口和文件分工

| 文件 | 负责什么 |
| --- | --- |
| [school_planner.py](isaac_snapshot/school_planner.py)、[step3a_planner.py](isaac_snapshot/step3a_planner.py) | 逆解、碰撞筛选、路径适配与绕行 |
| [step1_mug_pairs.py](isaac_snapshot/step1_mug_pairs.py)、[tier1_plan.py](isaac_snapshot/tier1_plan.py) | 杯身／杯柄示教模板和第一批采集计划 |
| [step3a_assets.py](isaac_snapshot/step3a_assets.py)、[step3a_rays.py](isaac_snapshot/step3a_rays.py)、[part_surface.py](isaac_snapshot/part_surface.py) | 资产准入、孔洞物理核查、指垫接触归属 |
| [step2_dev.py](isaac_snapshot/step2_dev.py)、[step3a_verify.py](isaac_snapshot/step3a_verify.py) | 分片录制、原指令回放、配对验收 |
| [run_lab_school.sh](isaac_snapshot/run_lab_school.sh)、[lab_school_probe.py](isaac_snapshot/lab_school_probe.py) | 启动 Isaac Sim，加载分片并写回执，隔离 Kit 缓存 |
| [unified_grasp.py](isaac_snapshot/unified_grasp.py)、[episode_store.py](isaac_snapshot/episode_store.py) | 统一控制、试提判分和对齐 HDF5 写入 |
| [step3b_control.py](isaac_snapshot/step3b_control.py)、[step3b_common.py](isaac_snapshot/step3b_common.py) | 总控、进度、资源租约、暂停与续跑 |
| [step3b_collect.py](isaac_snapshot/step3b_collect.py) | 扩采配额、计划、失败重试、验收与合库 |
| [step3b_train.py](isaac_snapshot/step3b_train.py)、[step3b_eval.py](isaac_snapshot/step3b_eval.py) | 训练 V3-A、开发集选点、冻结评测和最终报告 |

其他快照脚本的用途逐文件列在 [仿真快照文件索引](isaac_snapshot/INDEX.md)。日常交接先从表中的主入口定位。需要替换模型时，沿用专家与 HDF5 接口，并替换策略前向和仿真端动作适配。
