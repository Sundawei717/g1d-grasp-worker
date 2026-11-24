"""Render the completed, pinned evaluation without changing its measurements."""
import csv
import json
from pathlib import Path

ROOT = Path('/data/sundawei/g1d_grasp_worker')
BASE = ROOT/'runs/step3b_01_gripperfix_a40_dev28_20261124'


def read(path):
    return json.loads(path.read_text())


def joint(row):
    return row['status']=='success' and row['part_hit'] is True


def main():
    data=read(BASE/'result.json')
    audit=read(BASE/'physical_audit.json')
    old={r['case']:r for r in read(ROOT/'runs/v3_step2_followup_01/a40_last_dev.json')['episodes']}
    rows=data['episodes']
    assert len(rows)==len(old)==28 and {r['case'] for r in rows}==set(old)
    s=data['strata']['all']; a=s['before']; b=s['after']
    labels={'mug':'杯子','basket':'篮子','carton':'长盒','mug_body':'杯身','mug_handle':'杯柄',
            'basket_handle':'篮提手','basket_rim':'篮沿','box_negative_end':'长盒负端','box_positive_end':'长盒正端',
            'obstacle':'有障碍','non_obstacle':'无障碍'}
    terminal={'success':'抓起','timeout':'超时','lift_failed':'试提失败'}
    gains=[r['case'] for r in rows if r['status']=='success' and old[r['case']]['status']!='success']
    losses=[r['case'] for r in rows if r['status']!='success' and old[r['case']]['status']=='success']
    hit_gains=[r['case'] for r in rows if joint(r) and not joint(old[r['case']])]
    hit_losses=[r['case'] for r in rows if not joint(r) and joint(old[r['case']])]
    correct=s['after_correct']
    lines=['# A40 last.pt：夹爪修复后 28 开发场景复测','',
           f"完成时间（UTC）：{data['completed_at']}。本次在 114 的 GPU 0、2 上运行，结果保存在 115。",'',
           f"抓起 **{a['success']}/28 → {b['success']}/28**（净变化 {b['success']-a['success']:+d} 场）；抓起且部位正确 **{s['before_correct']}/28 → {correct}/28**（净变化 {correct-s['before_correct']:+d} 场）。",
           '本轮结论：开口指令的大幅跳变已消除，整体抓起率没有提高，抓起且部位正确少了 1 场。','',
           '使用同一个 A40 last.pt，不训练；固定原 28 场景、初态计划、种子、首帧框与抓取点。此次 28 场全部重新运行，没有沿用此前 3 场探针结果。',
           '保留原开发集的固定障碍设置，避免把动态障碍混入夹爪修改的对比。统一同步 20 Hz、15 秒/300 步、8 cm 垂直试提、0.55 秒悬停、TE exec=1/m=0.01。','',
           '## 总成绩','',
           '| 指标 | 修复前 | 修复后 |', '|---|---:|---:|',
           f"| 抓起 | {a['success']}/28 ({a['success']/28:.1%}) | {b['success']}/28 ({b['success']/28:.1%}) |",
           f"| 抓起且部位正确 | {s['before_correct']}/28 ({s['before_correct']/28:.1%}) | {correct}/28 ({correct/28:.1%}) |",
           f"| 抓起但部位不正确 | {a['success']-s['before_correct']} | {b['success']-correct} |",
           f"| 超时 | {a['timeout']} | {b['timeout']} |",
           f"| 试提失败 | {a['lift_failed']} | {b['lift_failed']} |",'',
           '“抓起且部位正确”同时要求 success 与 Part Hit=true，统一以 28 场作分母；接触部位正确但试提失败不算成功。','',
           f"抓起结果：改善 {len(gains)} 场（{', '.join(gains) or '无'}）；退步 {len(losses)} 场（{', '.join(losses) or '无'}）。",
           f"抓起且命中结果：改善 {len(hit_gains)} 场（{', '.join(hit_gains) or '无'}）；退步 {len(hit_losses)} 场（{', '.join(hit_losses) or '无'}）。",'',
           '## 分层结果','',
           '| 分层 | 场数 | 抓起：旧→新 | 抓起且部位正确：旧→新 |', '|---|---:|---:|---:|']
    for key in ('mug','mug_body','mug_handle','basket','basket_handle','basket_rim','carton','box_negative_end','box_positive_end','obstacle','non_obstacle'):
        v=data['strata'][key]
        lines.append(f"| {labels[key]} | {v['before']['count']} | {v['before']['success']}→{v['after']['success']} | {v['before_correct']}→{v['after_correct']} |")
    lines += ['', '杯/篮/长盒相加为 28；部位和是否有障碍是同一批场景的不同切法，不能把所有行相加。','',
              '## 跳变修复与失败诊断','']
    x,y=(audit[k]['summary'] for k in ('before','after'))
    lines += [f"- 实际下发开口单步增加超过 4.01 mm：**{x['jump_events']} 次/{x['jump_episodes']} 场 → {y['jump_events']} 次/{y['jump_episodes']} 场**；最大单步增加 {x['max_upward_mm']:.2f} → {y['max_upward_mm']:.2f} mm。该统计包括所有较大开口增加，不能把旧记录每一次都认定为同一个 Bug。",
              '- C_6491：旧运行第 105 步开口突然增加约 5.99 mm，最长仅连续稳定 2 步并超时；本次没有这类跳变，第 110 步触发试提，目标在悬停期最低抬高约 79.9 mm，成功且命中。',
              f"- 实际执行抬升的回合数：{x['physical_lift_trials']} → {y['physical_lift_trials']}。空夹不再触发无意义试提，但若最终仍未夹住，只是由预检失败改为超时，不计作成功收益。",
              f"- `lost_grip_during_lift_or_hold`：{x['loss_during_lift_or_hold']} → {y['loss_during_lift_or_hold']} 场。物理轨迹中“脱离支撑、目标参考点升高至少 10 mm，随后落回支撑面且失去双指接触”：{x['airborne_ge_10mm_then_supported_without_two_pads']} → {y['airborne_ge_10mm_then_supported_without_two_pads']} 场。这是辅助诊断，不替换原来的悬停成功判定。",
              '- C_9815：第 107 步触发试提后约 0.135 s 失去双指接触；目标参考点最高仅上移约 1.89 mm，未稳定带离桌面。触发时夹紧余量约 6.29 mm，双指垫约 22.25/19.72 N，已超过旧版 6 mm 和 5 N 要求，不能单凭本例认定是余量阈值改为 2 mm 导致。',
              f"- 接近期目标位移 ≥10 mm：{a['disturbance']['target_translation_ge_10mm']} → {b['disturbance']['target_translation_ge_10mm']} 场；倾斜 ≥15°：{a['disturbance']['target_tilt_ge_15deg']} → {b['disturbance']['target_tilt_ge_15deg']} 场。",'',
              '本次是与历史 28 场结果的一次成对复测。初态、种子和目标姿态通过核对，但渲染图像不是逐像素相同，例如 C_9818 首步关节目标已差约 0.064°。因此本报告说明本轮净变化，不能单凭个别成败翻转证明因果收益。', '',
              '## 28 场逐条对照','',
              '| 场景 | 指定部位 | 障碍 | 旧结果 | 新结果 | 旧→新 Part Hit | 新控制步数 | 新诊断 |',
              '|---|---|---|---|---|---|---:|---|']
    csvrows=[]
    detailed={r['case']:r for r in audit['after']['episodes']}
    for r in rows:
        p=old[r['case']]
        if r['status']=='success':
            reason='部位正确' if r['part_hit'] else '抓起但抓错部位'
        elif r['failure']=='lost_grip_during_lift_or_hold':
            reason='试提中未保持稳定夹持'
        elif r['failure']=='lift_path_rejected':
            reason='抬升安全预检未通过'
        else:
            q=detailed[r['case']]
            reason='没有建立满足条件的双指夹持' if q['bilateral_loaded_control_steps']==0 else '未持续满足全部试提条件'
        lines.append(f"| {r['case']} | {labels.get(r['part'],r['part'])} | {'有' if r['obstacle'] else '无'} | {terminal[p['status']]} | {terminal[r['status']]} | {p['part_hit']}→{r['part_hit']} | {r['steps']} | {reason} |")
        csvrows.append({'case':r['case'],'asset':r['bucket'],'part':r['part'],'obstacle':r['obstacle'],
                        'before_status':p['status'],'after_status':r['status'],'before_part_hit':p['part_hit'],
                        'after_part_hit':r['part_hit'],'steps':r['steps'],'failure':r['failure'],'diagnostic':r['path']})
    lines += ['', '## 本轮执行器版本','',
              '- 预紧根据请求开口相对实测开口判断，不再要求请求相对上一指令单调减小；下发指令变化限于每控制步 ±4 mm。',
              '- 闭爪意图采用相对开口判断，取代固定 71 mm 尺寸门槛；夹紧余量上限要求改为 2 mm，并受剩余机械行程的一半限制。双指垫仍各需至少 5 N。',
              '- 保持闭合意图 0.25 s、手臂稳定窗口 0.20 s、开口变化 <0.5 mm 连续 0.15 s；空夹触底本身不能触发试提。',
              '- 试提仍锁定最后下发的夹爪目标，成功判定和 Part Hit 几何判定未改变。','',
              '## 文件与复现','',
              f"- 完整结果：`{BASE}/result.json`。",
              f"- 控制跳变与物理轨迹审计：`{BASE}/physical_audit.json`。",
              f"- 逐场表格：`{BASE}/comparison.csv`。",
              f"- 完成标志：`{BASE}.complete`（PASS 表示 28 场评测和配对校验完成）。",
              f"- 模型 SHA-256：`{data['contract']['checkpoint']['sha256']}`。",
              f"- 开发清单 SHA-256：`{data['contract']['development_sha256']}`。",
              '- 原始逐步诊断路径见结果 JSON 的 `episodes[].path`，同目录有 `physics_trace.jsonl`。','']
    (BASE/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    with (BASE/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]))
        w.writeheader();w.writerows(csvrows)
    print(json.dumps({'success_before':a['success'],'success_after':b['success'],
                      'correct_before':s['before_correct'],'correct_after':correct,
                      'gains':gains,'losses':losses,'correct_gains':hit_gains,'correct_losses':hit_losses,
                      'before_failures':a['failure_types'],'after_failures':b['failure_types'],
                      'physical_audit':{'before':x,'after':y}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
