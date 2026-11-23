# 第一次用专家脚本采集

专家脚本让 G1D 在 Isaac Sim 里按指定部位抓取，然后录下画面、机器人状态和动作。**先录两条看看效果**，用 [record_expert_demo.py](record_expert_demo.py)。**想一次采上百条不同任务**，看 [批量采集指南](batch-collection.md)。

## 录一组你想看的场景

在已配置仿真的实验服务器上，进入从 GitHub 下载的仓库根目录。下例录 **小篮子：抓顶部提手、抓侧边篮沿**。

```bash
/data/sundawei/vla_rsi_g1d/envs/sim60/bin/python simulation/record_expert_demo.py --project-dir /data/sundawei/g1d_grasp_worker --preset basket
```

换物体或场景，只改最后的 `basket`。下表每个预设录两次，分别验证抓起和部位命中。

| 命令中的预设 | 会录什么 |
| --- | --- |
| `mug` | 棕色带柄杯，−15°，同一只杯子分别夹杯身、杯柄 |
| `mug-minus75` | 棕杯转到 −75°，分别夹杯身、杯柄 |
| `mug-plus75-obstacle` | 棕杯转到 +75°，桌上有挡路物，两种部位都绕过去夹 |
| `basket` | 小篮 0°，夹顶部提手、侧边篮沿 |
| `basket-plus75` | 小篮转到 +75°，夹提手、篮沿 |
| `basket-obstacle` | 小篮 +34°，有挡路物，夹提手、篮沿 |
| `long-box` | 长盒 −85°，分别夹物体长轴两端 |
| `long-box-obstacle` | 长盒 +52°，绕过挡路物后分别夹两端 |
| `white-box-pepsi` | 同一张桌子上有白盒和百事罐，一次抓白盒，一次抓罐。沿用已验收的原专家动作重录 |
| `sugar-box`、`ycb-mug`、`raised-rim-basket` | 糖盒两端、YCB 带柄杯杯身／杯柄、高侧沿篮提手／篮沿 |
| `marker-long-box`、`compact-box`、`round-package`、`matte-soda` | 竖放标记长盒、紧凑盒、圆包装、哑光罐。对同一部位分别录无障与绕障 |

偏航角是物体在桌面上转过的角度。正负表示转向。上述预设指向已经验真过的场景，可以用 `python simulation/record_expert_demo.py --list` 列出完整名称。实际采集命令末尾加 `--check` 会先检查配置和专家计划。去掉 `--check` 才会启动仿真。使用另一台实验机时，换成本机的仿真 Python 和工程目录。 [安装与资产交接](operations.md#运行环境与交接文件)

## 看录制结果

- 终端出现 `PASS: 2/2` 表示两条轨迹都录制并重放成功，且夹到了要求的部位。
- 终端会打印每条 `.h5` 和验收 JSON 的完整路径。HDF5 保存头部、左腕、右腕画面及实际动作。
- 示例用于验证环境和熟悉流程。正式扩采会生成新初态和新 ID，然后按配对整组入库。

想了解新角度、多物换目标和障碍怎样批量生成，见 [批量采集指南](batch-collection.md)。专家规划和回放逻辑分别位于 [棕杯专家](isaac_snapshot/step1_mug_pairs.py)、[通用规划器](isaac_snapshot/step3a_planner.py) 和 [录制验收器](isaac_snapshot/step3a_verify.py)。

[返回仓库首页](../README.md) · [全部脚本索引](isaac_snapshot/INDEX.md)
