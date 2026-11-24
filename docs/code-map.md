# 数采、训练和评测代码入口

[返回首页](../README.md) · [数据管线](data-pipeline.md) · [项目进展](experiments.md)

这份导航对应 2026-11-24 的源码同步。`simulation/isaac_snapshot/` 保存实验服务器的平铺脚本，`src/g1d_grasp/` 是便于换模型的可安装数据与模型接口。

## 先找这几个入口

| 要做的事情 | 入口 | 配套说明 |
|---|---|---|
| 先采一组杯、篮、盒、罐或绕障示教 | [record_expert_demo.py](../simulation/record_expert_demo.py) | [专家采集入门](../simulation/README.md) |
| 批量生成、录制、回放、验收和合库 | [step3b_collect.py](../simulation/isaac_snapshot/step3b_collect.py) | [批量采集](../simulation/batch-collection.md) |
| 串联数采 → 训练 → 评测，查看进度及暂停/恢复 | [step3b_control.py](../simulation/isaac_snapshot/step3b_control.py) | [运行环境和运维](../simulation/operations.md) |
| 读取 HDF5、按组划分、取模型训练窗口 | [src/g1d_grasp/data](../src/g1d_grasp/data/) | [数据管线与换模型](data-pipeline.md) |
| 当前 V3-A 的 80 轮正式训练 | [step3b_train.py](../simulation/isaac_snapshot/step3b_train.py) | 全 Epoch、单卡累积/双卡 DDP、续训、best/last/逐轮曲线 |
| 实验使用的网络、监督和输入组装 | [chunk_model.py](../simulation/isaac_snapshot/chunk_model.py)、[three_view_train.py](../simulation/isaac_snapshot/three_view_train.py)、[v3_prompt_train.py](../simulation/isaac_snapshot/v3_prompt_train.py) | [模型结构与输入输出](model-v3a.md) |
| 28 开发场景选点、80 冻结任务对照和报告 | [step3b_eval.py](../simulation/isaac_snapshot/step3b_eval.py) | 同版执行器复测 A40 基线，按合同校验缓存 |
| 在线模型执行与抓取判分 | [three_view_rollout.py](../simulation/isaac_snapshot/three_view_rollout.py)、[unified_grasp.py](../simulation/isaac_snapshot/unified_grasp.py)、[strict_grasp.py](../simulation/isaac_snapshot/strict_grasp.py) | [夹爪版本与试提](../simulation/gripper-control.md) |

## 数采依赖也在仓库里

专家规划用 `school_planner.py`、`step3a_planner.py` 和各资产模板；资产准备包括 `prepare_assets.py`、`step1_mug_asset.py`、`step1b_asset.py` 与 `step3a_assets.py`。录制/回放通过 `lab_school_probe.py`、`rigid_grasp_probe.py`、`step2_dev.py` 和 `step3a_verify.py`。无损 HDF5 由 `step1_episode.py` 写入，`tier1_collect.py` 校验入库，`step3b_collect.py` 合库与切分。

本次补齐了此前漏收的资产检查、`tier1_finalize.py`、前向性能诊断、训练/审计启动脚本，以及服务器原始测试文件。每个文件的职责见 [完整脚本索引](../simulation/isaac_snapshot/INDEX.md)。早期 B 组检测缓存和旧训练实验源码也保留了，但当前定型路线是 V3-A。

相机安装、同时间戳取图、渲染和 Isaac Sim 6.0 兼容桥的五个项目模块位于 [simulation/runtime](../simulation/runtime/README.md)。它们原先位于另一个本地工程中，本次一并收录，避免交接时只带主脚本却缺少桥接源码。

## 仓库版本与生产版本

仓库已包含最新评测修复：明确使用 `adaptive` 夹爪、按源代码/计划/提示/标定绑定缓存、重测同版 A40 开发基线，以及补齐抬升/悬停碰撞统计。

**发布源码不等于替换运行中的进程。** 截至本次同步，生产总控仍在采集，修复位于独立暂存目录；正式部署应在安全边界进行，并重新执行预检、重启总控。旧数据和旧报告不会因为 GitHub 更新而自动改口径。详见 [工程审查报告](reports/2026-11-24-pipeline-audit.md)。

[源码清单](../simulation/source_manifest.json) 记录逐文件 SHA-256（统一 LF 后计算）、来源与生产快照对照。`review_patch` 表示经过审查测试的新代码；`production_snapshot` 表示与服务器源文件一致。

## 检查与数据交接

在仓库根目录运行：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

默认测试目录是 `tests/`。`simulation/isaac_snapshot/test_*.py` 是随历史服务器源码归档的测试，部分依赖旧实验、外部权重或仿真环境，不代表已经在本次同步中全部执行。

真实 HDF5、模型权重、物体/机器人资产、相机标定、完整场景清单和运行产物在数据盘单独交接。GitHub 中包含处理这些内容的代码、身份清单和说明，不能仅凭克隆仓库启动完整 Isaac Sim 实验。
