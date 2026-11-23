# 仿真快照文件索引

[返回专家采集与运行指南](../README.md)

以下按用途定位快照源文件。部署条件和现行运行命令在上面的指南。历史入口依赖各自的旧版清单和回执。

## 现行扩采、训练与评测

| 文件 | 职责 |
| --- | --- |
| [step3b_control.py](step3b_control.py) | 启动、状态、暂停、续跑与三阶段完成回执。 |
| [step3b_common.py](step3b_common.py) | 固定清单、原子进度写入、GPU 资源租约与 Kit 清理。 |
| [step3b_collect.py](step3b_collect.py) | 分配新场景、生成专家计划、录制回放及验收合库。 |
| [step3b_train.py](step3b_train.py) | 80 轮无放回优化、梯度累积和 DDP、检查点、续训与损失曲线。 |
| [step3b_eval.py](step3b_eval.py) | 用开发集选终选权重、跑冻结终榜和输出报告。 |
| [background_status.py](background_status.py) | 查看后台运行、退出码及完成标志。 |

## 仿真、执行器和数据底层

| 文件 | 职责 |
| --- | --- |
| [run_lab_school.sh](run_lab_school.sh) | 设置 Linux 仿真环境、运行分片并清理分片缓存。 |
| [lab_school_probe.py](lab_school_probe.py) | 载入机器人、桌面、物体与任务清单，驱动 Isaac Sim。 |
| [school_planner.py](school_planner.py) | 生成抓取候选腕姿并做逆运动学、碰撞及路径筛选。 |
| [rigid_grasp_probe.py](rigid_grasp_probe.py) | 执行物理抓取，采集机器人反馈、接触、日志和图像。 |
| [unified_grasp.py](unified_grasp.py) | 20 Hz 闭环、试提门控、回合时间和终态。 |
| [vertical_lift.py](vertical_lift.py) | 求解并筛选垂直抬升路径。 |
| [strict_grasp.py](strict_grasp.py) | 按物体姿态、接触与搬运状态计算抓取验收指标。 |
| [grasp_regions.py](grasp_regions.py) | 抓取部位名称及区域定义。 |
| [part_surface.py](part_surface.py) | 从原网格图谱、接触点和碰撞几何判定部位。 |
| [episode_store.py](episode_store.py) | 写入示教及回放所需的时间对齐数据。 |
| [dataset_contract.py](dataset_contract.py) | 早期数据布局和一致性检查。 |
| [chunk_model.py](chunk_model.py) | 仿真实验用动作块网络和模型损失辅助函数。 |
| [three_view_inputs.py](three_view_inputs.py) | 仿真侧三视角、提示几何和状态编码。 |
| [prompt_views.py](prompt_views.py) | 提示掩码、热图与图像变换。 |
| [three_view_train.py](three_view_train.py) | 三视角窗口加载和原版训练入口。 |
| [three_view_rollout.py](three_view_rollout.py) | 加载三视角策略并组装在线本体、相机和提示输入。 |
| [three_view_eval.py](three_view_eval.py) | 准备、冻结和比较三视角闭环评测。 |
| [temporal_actions.py](temporal_actions.py) | 按时间对齐并融合重叠动作块。 |
| [policy_rollout.py](policy_rollout.py) | 较早的策略接近流程和提示输入实验入口。 |
| [train_policy.py](train_policy.py) | 较早的训练及检查点工具。 |
| [inspect_environment.py](inspect_environment.py) | 检查仿真配置和依赖路径。 |

## 前期专家示教、Tier-1 和统一执行器验收

| 文件 | 职责 |
| --- | --- |
| [three_view_step0.py](three_view_step0.py) | 准备并汇总统一执行器的专家和旧权重回放体检。 |
| [run_three_view_step0.sh](run_three_view_step0.sh) | 执行 Step 0 子任务和汇总。 |
| [step1_mug_pairs.py](step1_mug_pairs.py) | 准备棕杯杯身／杯柄配对、组织回放和验收。 |
| [step1_episode.py](step1_episode.py) | Step 1 小样本的轨迹落盘实现。 |
| [step1b_asset.py](step1b_asset.py) | 制作提手篮等首批扩展资产和相关几何。 |
| [step1b_scenes.py](step1b_scenes.py) | 验证篮、宽偏航、长盒、绕障及统一门控。 |
| [tier1_plan.py](tier1_plan.py) | 生成 Tier-1 专家计划、配对与开发候选。 |
| [tier1_collect.py](tier1_collect.py) | 处理旧示教重录、物理分片、验收回执和 Tier-1 入库。 |
| [run_tier1_v3.sh](run_tier1_v3.sh) | 运行 Tier-1 各采集准备及验收环节。 |
| [step2_dev.py](step2_dev.py) | 准备并验真开发场景，提供共用分片执行接口。 |

## Tier-2 资产准入和保留测试集

| 文件 | 职责 |
| --- | --- |
| [step3a_assets.py](step3a_assets.py) | 清点基础网格、准备凸分解与部位图谱。 |
| [step3a_fetch_ycb.sh](step3a_fetch_ycb.sh) | 导入项目用的 YCB 资产。 |
| [step3a_rays.py](step3a_rays.py) | 用网格和 PhysX 射线检查把手、篮筐孔洞。 |
| [step3a_planner.py](step3a_planner.py) | 按已标注部位与新场景生成配对或绕障规划。 |
| [step3a_verify.py](step3a_verify.py) | 小样本录制、原生动作回放、Part Hit 和配对验收。 |
| [step3a_suites.py](step3a_suites.py) | 构造 L2 至 L4 测试场景和 Tier-2 小样本计划。 |
| [step3a_l1.py](step3a_l1.py) | 从未使用计划构建并验真同分布 L1。 |
| [step3a_freeze.py](step3a_freeze.py) | 在专家验真后冻结 80 个任务和网格指纹。 |
| [step3a_pilots_parallel.py](step3a_pilots_parallel.py) | 并行核验新增资产的多组小样本。 |
| [step3a_preplan.py](step3a_preplan.py) | 在物理回放期间预生成候选计划。 |
| [step3a_probe_plan.py](step3a_probe_plan.py) | 检查资产候选轨迹和腕部姿态规划。 |
| [step3a_basket_calibration.py](step3a_basket_calibration.py) | 校准篮子抓取模板和抬升。 |
| [step3a_sugar_calibration.py](step3a_sugar_calibration.py) | 按糖盒实际支撑位姿重算配对。 |
| [step3a_compact_calibration.py](step3a_compact_calibration.py) | 校准紧凑盒接触位置和部位边界。 |
| [step3a_marker_detour.py](step3a_marker_detour.py) | 筛选竖放标记长盒的挡路及绕障方案。 |
| [step3a_marker_followup.py](step3a_marker_followup.py) | 补验标记长盒独立绕障组。 |
| [step3a_round_followup.py](step3a_round_followup.py) | 补验圆包装的第二组布局。 |
| [step3a_local_guides.py](step3a_local_guides.py) | 利用已成功专家腕姿引导替代资产。 |
| [step3a_local_substitutes.py](step3a_local_substitutes.py) | 验证本地替代资产的可抓取性。 |
| [step3a_final_substitutes.py](step3a_final_substitutes.py) | 对剩余替代资产逐个做最终小样本筛选。 |
| [step3a_guide_diagnostics.py](step3a_guide_diagnostics.py) | 检查腕姿引导和筛选失败反馈。 |
| [step3a_resources.py](step3a_resources.py) | 限制 Step 3-A 并行仿真资源占用。 |
| [step3a_memory_watch.py](step3a_memory_watch.py) | 监控 Step 3-A 仿真批次的主机内存。 |
| [step3a_switch_gpu.py](step3a_switch_gpu.py) | 等待旧分片结束后迁移到 GPU 租约调度。 |
| [step3a_report.py](step3a_report.py) | 汇总新增资产、冻结场景及 A40／B40 对照。 |
| [resume_step3a_pilots.py](resume_step3a_pilots.py) | 从已落盘回执接续小样本验收。 |
| [resume_step3a_suite.sh](resume_step3a_suite.sh) | 恢复 Step 3-A 场景套件。 |
| [run_step3a_report_watch.sh](run_step3a_report_watch.sh) | 定期生成 Step 3-A 进度报告。 |

## 模型诊断、A/B 和历史检测实验

| 文件 | 职责 |
| --- | --- |
| [step2_eval.py](step2_eval.py) | 执行 V3-A、V3-B 的训练场景及开发集闭环对照。 |
| [step2_report.py](step2_report.py) | 汇总 Step 2 损失、闭环结果和开发验真。 |
| [step2_teacher_diagnostic.py](step2_teacher_diagnostic.py) | 离线专家观测上的开环预测和训推一致性体检。 |
| [v3_prompt_repair.py](v3_prompt_repair.py) | 提示通路修复后的配对训练与通关验证。 |
| [v3_prompt_train.py](v3_prompt_train.py) | 调整提示学习率、增强、L1 和无放回采样的训练入口。 |
| [v3_a_micro.py](v3_a_micro.py) | 两组 V3-A 小规模训练的闭环和时序融合对照。 |
| [v3_a_micro_forensics.py](v3_a_micro_forensics.py) | 分析原 10 条配对闭环的分叉、偏离和提示响应。 |
| [v3_query_quickcheck.py](v3_query_quickcheck.py) | 训练前复核自碰诊断和关键场景拟合。 |
| [v3_query_gate.py](v3_query_gate.py) | 验证完整框和点直注动作 Query 后的闭环闸门。 |
| [v3_followup_fit_watch.py](v3_followup_fit_watch.py) | 按内存预算安排 B20 与 A40 两组训练。 |
| [v3_followup_gpu.py](v3_followup_gpu.py) | 按主机 RAM 和 GPU 显存调度历史闭环分片。 |
| [v3_followup_eval.py](v3_followup_eval.py) | 对照 B20、A40 及原 A20 的训练和开发成绩。 |
| [v3_b40_eval.py](v3_b40_eval.py) | 核查 B40 与 A40 的单变量隔离并汇总配对闭环。 |
| [reserve_b40_eval_gpu.sh](reserve_b40_eval_gpu.sh) | 为 B40 闭环管理可用 GPU。 |
| [run_v3_b40_fit.sh](run_v3_b40_fit.sh) | 启动历史 B40 训练。 |
| [run_v3_followup_fit.sh](run_v3_followup_fit.sh) | 启动 B20 和 A40 的后续对照。 |
| [run_v3_prompt_train.sh](run_v3_prompt_train.sh) | 启动提示修复版训练。 |
| [run_v3_query_fit.sh](run_v3_query_fit.sh) | 启动框和抓取点直注 Query 的训练。 |
| [audit_fit20_supervision.py](audit_fit20_supervision.py) | 核对早期训练中接近、闭爪和抬升监督范围。 |
| [three_view_detector.py](three_view_detector.py) | 早期腕部检测与缓存相关工具。 |
| [v3_detector.py](v3_detector.py) | V3-B 检测配置、推理和版本化缓存。 |
| [step1_detector.py](step1_detector.py) | 右腕候选检测器的早期离线摸底。 |
| [step1b_detector.py](step1b_detector.py) | 补测轻量及半精度检测器的精度、延迟和闪烁。 |
| [benchmark_tracking.py](benchmark_tracking.py) | 对比不同视觉跟踪方案。 |
| [tracking.py](tracking.py) | 实现早期提示跟踪和跟踪器适配。 |
| [tracking_motion.py](tracking_motion.py) | 构建目标运动场景及跟踪评估。 |
| [run_tracking_benchmark.sh](run_tracking_benchmark.sh) | 运行离线跟踪对照。 |
| [run_tracking_motion.sh](run_tracking_motion.sh) | 采集并测试含目标运动的跟踪场景。 |
| [setup_sam2_tracking.sh](setup_sam2_tracking.sh) | 准备早期跟踪实验的 SAM2 环境。 |
| [prompt_mode_comparison.py](prompt_mode_comparison.py) | 比较可见真值、冻结等历史提示规则。 |
| [run_prompt_mode_comparison.sh](run_prompt_mode_comparison.sh) | 调度并汇总提示规则实验。 |
| [inspect_route_visibility.py](inspect_route_visibility.py) | 分析专家路径及抓取锚点的可见性。 |
| [run_visibility_replay.sh](run_visibility_replay.sh) | 复现可见性检查的专家轨迹。 |

## 旧版采集、回放与闭环实验

| 文件 | 职责 |
| --- | --- |
| [aggregate_grouped.py](aggregate_grouped.py) | 合并分片回执并统计专家场景。 |
| [prepare_grouped.py](prepare_grouped.py) | 生成旧版分组专家任务与计划。 |
| [screen_grouped_starts.py](screen_grouped_starts.py) | 筛除无法起动或无法到达的初始状态。 |
| [repair_grouped.py](repair_grouped.py) | 生成旧版专家批次失败项的替代计划。 |
| [replay_grouped_obstacle.py](replay_grouped_obstacle.py) | 构造并回放早期挡路物场景。 |
| [run_grouped_expert.sh](run_grouped_expert.sh) | 启动旧版专家分组采集。 |
| [run_grouped_obstacle_replay.sh](run_grouped_obstacle_replay.sh) | 跑旧版绕障专家回放并总结。 |
| [grouped_video.py](grouped_video.py) | 把分组仿真录像整理为演示视频。 |
| [collect_training_overnight.py](collect_training_overnight.py) | 按夜间预算分批采集早期训练数据。 |
| [run_collect_training_overnight.sh](run_collect_training_overnight.sh) | 托管历史夜间采集。 |
| [audit_overnight_data.py](audit_overnight_data.py) | 核对夜间数据质量、数量和回执。 |
| [audit_grasp_receipt.py](audit_grasp_receipt.py) | 核验单场抓取回执与保存字段。 |
| [prepare_collection_preview.py](prepare_collection_preview.py) | 准备快速采集预览的小批计划。 |
| [run_collection_preview.sh](run_collection_preview.sh) | 执行采集预览并汇总。 |
| [dataset_alignment.py](dataset_alignment.py) | 准备和汇总早期数据时间轴一致性验证。 |
| [run_dataset_alignment.sh](run_dataset_alignment.sh) | 调度早期对齐试验。 |
| [prepare_train_bootstrap.py](prepare_train_bootstrap.py) | 准备历史训练起步样本和回执。 |
| [run_train_bootstrap.sh](run_train_bootstrap.sh) | 录制并汇总早期训练样本。 |
| [prepare_policy_corrections.py](prepare_policy_corrections.py) | 从策略失败中构造专家纠正样本计划。 |
| [run_policy_correction_capture.sh](run_policy_correction_capture.sh) | 收集并审核纠正示教。 |
| [prepare_stage3_splits.py](prepare_stage3_splits.py) | 构造早期评测版本的分区清单。 |
| [three_view_collection.py](three_view_collection.py) | 准备、汇总早期三视角采集。 |
| [run_three_view_collection.sh](run_three_view_collection.sh) | 启动早期三视角采集并汇总。 |
| [three_view_diversity.py](three_view_diversity.py) | 增加早期多样场景的采集计划。 |
| [run_three_view_diversity.sh](run_three_view_diversity.sh) | 启动早期多样场景采集。 |
| [three_view_extra_plans.py](three_view_extra_plans.py) | 补充早期额外物理场景与抓取部位计划。 |
| [run_three_view_extra_physics.sh](run_three_view_extra_physics.sh) | 复跑早期额外物理场景。 |
| [run_three_view_parts_capture.sh](run_three_view_parts_capture.sh) | 采集早期多部位三视角样本。 |
| [three_view_scale.py](three_view_scale.py) | 扩大早期三视角采集规模并统计进度。 |
| [run_three_view_scale.sh](run_three_view_scale.sh) | 运行早期规模化采集。 |
| [run_three_view_pipeline.sh](run_three_view_pipeline.sh) | 串联旧版三视角采集、组织分区及评测。 |
| [run_three_view_generalization.sh](run_three_view_generalization.sh) | 运行历史三视角泛化对照。 |
| [three_view_time25.py](three_view_time25.py) | 在固定场景比较历史时间上限。 |
| [run_three_view_time25.sh](run_three_view_time25.sh) | 启动历史 25 秒基线对照。 |
| [run_three_view_training_data.sh](run_three_view_training_data.sh) | 组织早期三视角训练数据与训练流程。 |
| [fit20_eval.py](fit20_eval.py) | 组织早期 Fit20 权重的闭环评测。 |
| [fit20_retrain_comparison.py](fit20_retrain_comparison.py) | 对比早期可见性口径调整前后的重训结果。 |
| [run_fit20.sh](run_fit20.sh) | 启动早期 Fit20 拟合试验。 |
| [run_fit20_closed_loop.sh](run_fit20_closed_loop.sh) | 运行 Fit20 闭环评测。 |
| [run_fit20_corrected.sh](run_fit20_corrected.sh) | 运行使用纠正样本的 Fit20 试验。 |
| [run_fit20_occlusion_retrain.sh](run_fit20_occlusion_retrain.sh) | 运行早期遮挡口径重训对照。 |
| [run_fit20_residual.sh](run_fit20_residual.sh) | 运行早期残差策略试验。 |
