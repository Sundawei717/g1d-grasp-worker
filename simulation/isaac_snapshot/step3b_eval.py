"""Select with development episodes and evaluate the immutable final benchmark."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import step2_dev
from step2_eval import log_case, stats
from step3b_common import A40, DEV, FROZEN, FROZEN_SHA, RUNS, WORK, Resources, check_pause, now, read, save
from step3b_train import FIT
from three_view_detector import sha256
from three_view_eval import verify_policy
from unified_grasp import EPISODE_STEPS

EVAL = WORK / "evaluation"


def pinned_inputs():
    if sha256(FROZEN) != FROZEN_SHA:
        raise ValueError("Final benchmark manifest changed")
    dev = read(DEV)
    frozen = read(FROZEN)
    if (len(dev["cases"]) != 28 or not dev["replay_verified"] or frozen["count"] != 80
            or frozen["levels"] != {f"L{i}":20 for i in range(1,5)} or frozen["part_hit_passes"] != 80
            or EPISODE_STEPS != 300):
        raise ValueError("Evaluation inputs or executor changed")
    for row in dev["cases"] + frozen["cases"]:
        file = Path(row["plan"])
        plan = read(file)
        if not plan["feasible"] or plan["case"]["seed"] != row["seed"]:
            raise ValueError(f"Wrong deterministic plan {file}")
        if row.get("plan_sha256") and sha256(file) != row["plan_sha256"]:
            raise ValueError(f"Frozen plan changed: {file}")
        if not set(row["head_prompt"]) >= {"bbox_0","affordance_0"} or "expert_path" not in row:
            raise ValueError(f"Missing expert proof for {row['case_id']}")
    return dev["cases"], frozen["cases"]


def binding(folder, tag, pick, *, receipt_dir=None):
    result = read(folder/"result.json")
    path = folder/f"{pick}.pt"
    digest = sha256(path)
    cp = torch.load(path,map_location="cpu",weights_only=True)
    epoch, config = cp["epoch"], cp["config"]
    expected = result["best_epoch"] if pick=="best_val" else config["epochs"] if pick=="last" else int(pick.split("_")[1])
    if (epoch != expected or config != result["config"] or config.get("detector_config_sha256") is not None
            or config["variant"]!="A" or config["wrist_channels"]!=3 or not config["query_bbox"]
            or not config["point_queries"] or not config["freeze_bn"]):
        raise ValueError(f"Incompatible V3-A {tag} checkpoint: {path}")
    if pick == "last" and digest != result["last_sha256"]:
        raise ValueError("Changed final checkpoint")
    if pick == "best_val" and digest != result["checkpoint_sha256"]:
        raise ValueError("Changed best-validation checkpoint")
    out = (receipt_dir or EVAL)/f"{tag}_binding.json"
    save(out,{**result,"checkpoint":str(path),"checkpoint_sha256":digest})
    link = {"path":str(path),"receipt":str(out),"sha256":digest}
    verify_policy(link)
    return {**link,"epoch":epoch,"tag":tag}


def baseline_dev():
    raw = read(RUNS/"v3_step2_followup_01"/"a40_last_dev.json")
    if (len(raw["episodes"]) != 28 or raw["strata"]["all"]["success"] != 19
            or raw["success_and_hit"] != 16 or raw["m"] != .01):
        raise ValueError("Step 2 A40 dev anchor changed")
    cases = {r["case_id"]: r for r in read(DEV)["cases"]}
    return [{**r, "asset_family": cases[r["case"]]["asset_family"]} for r in raw["episodes"]]


def prepared(rows, tag):
    folder = EVAL/"plans"/tag
    folder.mkdir(parents=True,exist_ok=True)
    output = []
    for r in rows:
        plan = read(r["plan"])
        plan["case"]["te_decay"] = .01
        path = folder/f"{r['case_id']}.json"
        save(path,plan)
        output.append({**r,"plan":str(path),"te_decay":.01})
    return output


def evaluated(rows, tag, policy):
    check_pause()
    result = EVAL/f"{tag}.json"
    if result.exists():
        data = read(result)
        if data["checkpoint_sha256"] == policy["sha256"] and [r["case"] for r in data["episodes"]] == [r["case_id"] for r in rows]:
            return data["episodes"]
        raise ValueError(f"Changed policy or evaluation cases after {tag}")
    spec = prepared(rows,tag)
    step2_dev.WORK = WORK
    prompts = {r["case_id"]:r["head_prompt"] for r in spec}
    out = step2_dev.simulate(tag,spec,"three_view_closed_loop",policy=policy,prompts=prompts,detector=None,
                             resume_completed=True,allocator=Resources(),max_shard_cases=5,retries=64)
    episodes = []
    for row in spec:
        value = out[row["case_id"]]
        item = log_case(row,value,"a")
        if item["diagnostics"]["policy_execution"] != {"actions_per_decision":1,"temporal_decay":.01}:
            raise ValueError("Closed loop used the wrong execution contract")
        decisions = value[1]["policy_rollout"]["decisions"]
        if any(d["steps_applied"] != 1 or d["temporal_ensemble_chunks"] != min(8,i+1)
               for i,d in enumerate(decisions)):
            raise ValueError("Temporal ensemble audit failed")
        item.update(asset_family=row["asset_family"],tier=row.get("tier"),layout=row.get("layout_type"),
                    scene_id=row["scene_id"],pair_group_id=row.get("pair_group_id"))
        episodes.append(item)
    save(result,{"checkpoint_sha256":policy["sha256"],"episodes":episodes})
    return episodes


def joint(row):
    return row["status"]=="success" and row["part_hit"] is True


def paired(old,new):
    anchors = {r["case"]:r for r in old}
    if set(anchors) != {r["case"] for r in new} or len(anchors) != len(new):
        raise ValueError("Incomplete paired case coverage")
    score, score_joint = Counter(), Counter()
    detail = []
    for b in new:
        a=anchors[b["case"]]
        if a["seed"] != b["seed"] or a["head_prompt"] != b["head_prompt"]:
            raise ValueError(f"Changed seed or first-frame prompt: {b['case']}")
        x,y=np.asarray(a["settled_target"]),np.asarray(b["settled_target"])
        mm=float(1000*np.linalg.norm(x[:3]-y[:3]))
        q=float(abs(np.dot(x[3:],y[3:])/np.linalg.norm(x[3:])/np.linalg.norm(y[3:])))
        deg=float(np.rad2deg(2*np.arccos(np.clip(q,0,1))))
        for dest, (first, second) in ((score,(a["status"]=="success",b["status"]=="success")),
                                      (score_joint,(joint(a),joint(b)))):
            dest["both" if first and second else "tier1_only" if first else "tier2_only" if second else "neither"] += 1
        detail.append({"case":b["case"],"tier1":a["status"],"tier2":b["status"],
                       "tier1_success_and_hit":joint(a),"tier2_success_and_hit":joint(b),
                       "settled_translation_delta_mm":mm,"settled_rotation_delta_deg":deg})
    return dict(score),dict(score_joint),detail


def groups(old,new):
    strata = defaultdict(list)
    for r in new:
        labels = (["全部",r["asset_family"],r["part"],"有障碍" if r["obstacle"] else "无障碍"]
                  + ([f"L{r['tier']}"] if r.get("tier") else []) + ([r["layout"]] if r.get("layout") else []))
        for label in labels:
            strata[label].append(r)
    result = {}
    for label, vals in strata.items():
        keys = {r["case"] for r in vals}
        earlier = [r for r in old if r["case"] in keys]
        wins, wins_joint, details = paired(earlier,vals)
        result[label] = {"tier1":stats(earlier),"tier2":stats(vals),"paired":wins,"paired_joint":wins_joint,
                         "tier1_success_and_hit":sum(map(joint,earlier)),"tier2_success_and_hit":sum(map(joint,vals)),
                         "details":details}
    return result


def update(phase, **data):
    p=WORK/"stage3_progress.json"
    current=read(p,{})
    save(p,{**current,**data,"stage":3,"phase":phase,"updated_at":now()})


def report(tuning, winner, comparison, *, output=None):
    lines = ["# Tier-2 V3-A 扩量结果", "",
             f"终选第 {winner['epoch']} 轮 {winner['tag']}。开发集按抓起且命中数排序，其次按抓起数、轮次较早排序。",
             "终榜使用同步 20 Hz、15 秒／300 步、71 mm 门控、8 cm 试提和 0.55 秒悬停，TE exec=1、m=0.01。",
             f"冻结清单 SHA-256 {FROZEN_SHA}。", "", "## 28 个开发场景选点", "",
             "| 权重 | 轮次 | 抓起 | 抓起且命中 |", "|---|---:|---:|---:|",
             "| Tier-1 A40 last | 40 | 19/28 | 16/28 |"]
    for x in tuning:
        lines.append(f"| {x['tag']} | {x['epoch']} | {x['success']}/28 | {x['success_and_hit']}/28 |")
    lines += ["","## 冻结 80 任务逐层对照","",
              "指标按 Tier-1 → Tier-2 显示。Part Hit 的分母是可判定回合。配对按 Tier-1 独成／Tier-2 独成／同成／同败排列。","",
              "| 分层 | n | 抓起 | 抓起且命中 | Part Hit | lift_failed / timeout | 配对 | 目标位移 ≥10mm / 倾斜 ≥15° | 障碍位移 ≥10mm / 倾斜 ≥15° | 端到端 P50/P95 ms |",
              "|---|---:|---|---|---|---|---|---|---|---|"]
    for label,row in comparison.items():
        a,b=row["tier1"],row["tier2"]
        da,db=a["disturbance"],b["disturbance"]
        p=row["paired"]
        def latency(s):
            lat=s["latency_ms"].get("to_command_ms")
            return "—" if not lat else f"{lat['p50']:.1f}/{lat['p95']:.1f}"
        lines.append(f"| {label} | {a['count']} | {a['success']} → {b['success']} | {row['tier1_success_and_hit']} → {row['tier2_success_and_hit']} | "
                     f"{a['part_hit']}/{a['part_assessed']} → {b['part_hit']}/{b['part_assessed']} | "
                     f"{a['lift_failed']}/{a['timeout']} → {b['lift_failed']}/{b['timeout']} | "
                     + "/".join(str(p.get(k,0)) for k in ("tier1_only","tier2_only","both","neither")) + " | "
                     + f"{da['target_translation_ge_10mm']}/{da['target_tilt_ge_15deg']} → {db['target_translation_ge_10mm']}/{db['target_tilt_ge_15deg']} | "
                     + f"{da['fixture_translation_ge_10mm']}/{da['fixture_tilt_ge_15deg']} → {db['fixture_translation_ge_10mm']}/{db['fixture_tilt_ge_15deg']} | "
                     + f"{latency(a)} → {latency(b)} |")
    lines+=["","## 失败归因",""]
    for label,row in comparison.items():
        if label in ("全部","L1","L2","L3","L4"):
            def text(data):
                return "、".join(f"{k} {v}" for k,v in sorted(data.items())) or "无"
            lines.append(f"- {label}，Tier-1 {text(row['tier1']['failure_types'])}。Tier-2 {text(row['tier2']['failure_types'])}。")
    unsettled = [r for r in comparison["全部"]["details"]
                 if r["settled_translation_delta_mm"] > 1 or r["settled_rotation_delta_deg"] > 1]
    lines.extend(["",f"两模型复位后目标位姿相差超过 1 mm 或 1° 的场景 {len(unsettled)}/80，逐条误差见分层 JSON。"])
    lines+=["","[数据盘点](data_report.md) · [80 轮损失曲线](fit/loss_curves.png) · [分层 JSON](evaluation/final_comparison.json)"]
    output = output or WORK/"final_report.md"
    tmp=output.with_suffix(".tmp")
    tmp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    tmp.replace(output)


def evaluate():
    EVAL.mkdir(exist_ok=True)
    dev, frozen = pinned_inputs()
    base = binding(A40,"tier1_a40","last")
    anchor = baseline_dev()
    candidates = [binding(FIT,f"tier2_{k}",k) for k in ("epoch_60","last","best_val")]
    unique = {}
    for b in candidates:
        if b["epoch"] in unique:
            earlier = unique[b["epoch"]]
            lhs = torch.load(b["path"],map_location="cpu",weights_only=True)["model"]
            rhs = torch.load(earlier["path"],map_location="cpu",weights_only=True)["model"]
            if lhs.keys() != rhs.keys() or not all(torch.equal(lhs[k],rhs[k]) for k in lhs):
                raise ValueError("Different weights claim the same epoch")
        else:
            unique[b["epoch"]]=b
    tuning = []
    for b in unique.values():
        update("development",current=b["tag"],development=tuning)
        rows = evaluated(dev,f"{b['tag']}_dev",b)
        result = groups(anchor,rows)
        save(EVAL/f"{b['tag']}_dev_comparison.json",result)
        tuning.append({**b,"success":sum(r["status"]=="success" for r in rows),"success_and_hit":sum(map(joint,rows))})
    winner = max(tuning,key=lambda r:(r["success_and_hit"],r["success"],-r["epoch"]))
    selection=EVAL/"selection.json"
    chosen={"winner":winner,"development":tuning,"development_sha256":sha256(DEV)}
    if selection.exists() and read(selection) != chosen:
        raise ValueError("Tier-2 dev pick changed after final benchmark access")
    save(selection,chosen)
    if sha256(FROZEN) != FROZEN_SHA:
        raise ValueError("Frozen test digest changed before evaluation")
    left,right=[],[]
    for level in range(1,5):
        cases=[r for r in frozen if r["tier"]==level]
        update("frozen",level=level,completed_levels=level-1,current="tier1",winner=winner)
        left.extend(evaluated(cases,f"tier1_frozen_l{level}",base))
        update("frozen",level=level,completed_levels=level-1,current="tier2",winner=winner)
        right.extend(evaluated(cases,f"tier2_frozen_l{level}",winner))
    final=groups(left,right)
    save(EVAL/"final_comparison.json",{"winner":winner,"frozen_sha256":FROZEN_SHA,
         "strata":final,"tier1":left,"tier2":right})
    report(tuning,winner,final)
    update("complete",completed_levels=4,frozen_episodes=80,winner=winner,report=str(WORK/"final_report.md"))
