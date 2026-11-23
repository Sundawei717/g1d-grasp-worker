# Isaac Sim 采集与闭环

[isaac_snapshot](isaac_snapshot/) 保存了 Step 3-B 实际使用的控制器和本地脚本依赖快照。目录内保持运行时所需的平铺导入方式。目录根的可安装 Python 包供通用数据处理与训练使用。

| 环节 | 入口 | 用途 |
| --- | --- | --- |
| 采集调度 | isaac_snapshot/step3b_control.py | preflight、start、status、pause、resume |
| 录制与校验 | isaac_snapshot/step3b_collect.py | 专家规划、断点续采、8 cm 回放、Part Hit 验收、按组入库 |
| 训练参考 | isaac_snapshot/step3b_train.py | V3-A 80 轮、检查点续跑和 单卡／双卡 GPU 资源调度 |
| 闭环测试 | isaac_snapshot/step3b_eval.py | 28 开发场景选点、80 冻结场景终榜 |
| 仿真运行 | isaac_snapshot/run_lab_school.sh | Isaac Sim 启动、分片隔离与 Kit 缓存清理 |

该快照与原 Linux 环境绑定。运行时需要已有的 Isaac Sim 6.0 / Isaac Lab、G1D/Dex1 场景和机器人资产、校准后的相机和 URDF、安装在环境中的 projects.g1d_rsi 桥接模块、实际 env_resolved.json、前序验收回执及 runs 下的数据等。原脚本中有既有集群路径。恢复已有实例时按原目录结构部署。迁移新机时先校准资产路径与仿真环境，再运行 preflight。

当前工程做法是让仿真输出已验收的 HDF5 和训练清单，然后通过 g1d-data import-manifest 接入独立的数据包。替换模型只需要复用该数据接口。
