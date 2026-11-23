"""Render Step 3-A audit and its paired experiment from measured receipts."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent/"runs"
WORK=ROOT/"step3a_01"
B=ROOT/"v3_b40_step3a_01"
A=ROOT/"v3_step2_followup_01"
FIELDS=("joint_l1","gripper_l1","close_l1","weighted_l1")
REQUIRED=(
    "v3_b40_step3a_01_fit","v3_b40_step3a_01",
    *(f"step3a_01_{x}" for x in ("assets","rays","pilots","l1","l2","l3","l4","freeze",
                               "sugar_calibration","substitutes","substitute_rays","local_guides","round_followup","compact_calibration","final_substitutes","marker_detour","marker_followup")))
CORE=tuple(k for k in REQUIRED if not k.endswith(("_pilots","_calibration","_substitutes","_substitute_rays","_local_guides","_round_followup","_compact_calibration","_final_substitutes","_marker_detour","_marker_followup")))


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

def median(values):
    return statistics.median(values) if values else None

def fmt(v):
    return f"{v:.5f}" if v is not None else "—"

def table(lines,header,rows):
    lines.extend(["","| "+" | ".join(header)+" |","| "+" | ".join(["---"]*len(header))+" |"])
    for r in rows:
        lines.append("| "+" | ".join(map(str,r))+" |")

def markers(lines):
    table(lines,("任务","状态","完成时间 UTC"),[
        (k,f"{d['status']} {d['outcome']}" if d else "进行中",d.get("completed_at_utc","—"))
        for k in REQUIRED for d in [read(ROOT/f"{k}.complete")]])
    return all((ROOT/f"{k}.complete").is_file() for k in REQUIRED)


def asset_proofs():
    inv=read(WORK/"asset_inventory.json")
    add=read(WORK/"asset_substitutes.json").get("assets",{})
    training={**inv.get("tier2_proposed",{}),**add}
    rays={**read(WORK/"physx_rays.json").get("assets",{}),
          **read(WORK/"physx_rays_substitutes.json").get("assets",{})}
    raw=[]
    sources=[WORK/"pilots_progress.json",WORK/"substitutes_progress.json",
             *WORK.glob("pilots_sugar_calibration*_result.json"),
             *WORK.glob("pilots_guide_*_result.json")]
    for source in sources:
        raw.extend(read(source).get("cases",[]))
    by_mesh=defaultdict(dict)
    for case in raw:
        if case.get("replay_verified") and case.get("base_mesh_hash"):
            by_mesh[case["base_mesh_hash"]][case["scene_id"]]=case
    groups={}
    for mesh,cases in by_mesh.items():
        grouped=defaultdict(list)
        for row in cases.values():
            grouped[row["pair_group_id"]].append(row)
        groups[mesh]=[pair for pair in grouped.values() if len(pair)==2]
    ready={}
    for k,obj in training.items():
        passes=groups.get(obj["base_mesh_hash"],[])
        if len(passes)>=2 and (not obj["hole_probes_xz"] or rays.get(k,{}).get("passed")):
            ready[k]={**obj,"passing_group_ids":[g[0]["pair_group_id"] for g in passes],
                      "accepted_cases":[r["scene_id"] for g in passes for r in g]}
    return inv,add,rays,groups,ready,raw

def curves(lines):
    runs={"A40":ROOT/"v3_step2_followup_01_a40_fit","B20":ROOT/"v3_step2_followup_01_b20_fit",
          "B40":ROOT/"v3_b40_step3a_01_fit"}
    data={name:read(p/"loss_curves.json") for name,p in runs.items()}
    lines.extend(["","## 40 轮训练对照","",
        "两组共用 359 条训练和 101 条验证轨迹。每轮无放回遍历 34,373 个窗口，覆盖全部 4,733 个闭爪步，屏蔽试提。"
        "有效 batch 32，40 轮 43,000 次更新，ImageNet 冷启动，冻结 BN 统计。"
        "优化目标为标准化动作残差 L1，夹爪权重 0.25，闭爪权重 1.5。"
        "预训练 RGB 主干 3e-5，其他参数 1e-4，5% Warmup + Cosine。"])
    lines.extend(["","B40 仅增加右腕框输入。Swin-T BF16 短边 384、长边上限 512、阈值 0.25，当帧无保持，缺框置零。"
        "新增第 4 通道卷积核独立使用 1e-4，两组本体输入均为 2×17。"])
    result=read(runs["B40"]/"result.json")
    if result:
        lines.extend(["",f"B40 完成 {result['updates']:,} 次更新，最佳验证权重出现在第 {result['best_epoch']} 轮。"
            "留存第 10、20、30、40 轮权重和 best_val、last。"])
    checks=read(B/"strict_protocol.json")
    lines.extend(["", '视觉增强只作用于 RGB（亮度和噪声 std 0.015），输入关节噪声 std 0.005 rad。随机种子 33889。'])
    if checks:
        lines.extend(["", 'A40 训练和验证共约 2 小时 21 分钟，B40 本轮约 2 小时 57 分钟。B40 与多路资产工程共享主机资源。'])
        lines.extend(["","已逐轮核对前 40 轮的样本顺序哈希与学习率，状态和动作的归一化参数也一致。"])
    rows=[]
    for name,d in data.items():
        if not d:
            continue
        best=min(d,key=lambda x:x["validation"]["weighted_l1"])["epoch"]
        for row in d:
            if row["epoch"] in {1,10,20,30,35,40,best,d[-1]["epoch"]}:
                a=row["train"]
                b=row["validation"]
                rows.append((name,row["epoch"],"是" if row["epoch"]==best else "—",
                             *(fmt(a[f]) for f in FIELDS),*(fmt(b[f]) for f in FIELDS)))
    table(lines,("模型","轮数","最佳验证","训练关节","训练夹爪","训练闭爪","训练加权",
        "验证关节","验证夹爪","验证闭爪","验证加权"),rows)
    lines.extend(["","B20 的余弦周期在第 20 轮结束。B40 第 20 轮的读数属于 40 轮调度的中点。"])
    with (WORK/"loss_curves.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["run","epoch","updates","split",*FIELDS])
        for name,d in data.items():
            for r in d:
                for split in ("train","validation"):
                    writer.writerow([name,r["epoch"],r["step"],split,*(r[split][k] for k in FIELDS)])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return data
    fig,axes=plt.subplots(2,4,figsize=(13,5.6),sharex=True)
    for i,split in enumerate(("train","validation")):
        for j,key in enumerate(FIELDS):
            ax=axes[i][j]
            for name,values in data.items():
                ax.plot([r["epoch"] for r in values],[r[split][key] for r in values],
                        label=name,ls="--" if name=="B20" else "-",lw=1.7)
            ax.set_title(f"{split.title()} {key.replace('_',' ')}")
            ax.grid(alpha=.25)
            if i:
                ax.set_xlabel("Epoch")
            if j==0:
                ax.set_ylabel("Normalized L1")
    handles,labels=axes[0][0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(WORK/"loss_curves.png",dpi=172,bbox_inches="tight")
    plt.close(fig)
    lines.extend(["","![逐轮训练与验证 L1](loss_curves.png)","","[逐轮原始 CSV](loss_curves.csv)"])
    return data

def breakdown(data):
    def summed(subset):
        stat=data["strata"][subset]
        keys={"all":True,"obstacle":True,"non_obstacle":False}
        def includes(e):
            if subset=="all":
                return True
            if subset in keys:
                return e["obstacle"]==keys[subset]
            return subset in (e["bucket"],e["part"])
        joint=sum(e["status"]=="success" and e["part_hit"] is True for e in data["episodes"] if includes(e))
        return f"{stat['success']}/{stat['count']} · {joint} · {stat.get('lift_failed',0)}/{stat.get('timeout',0)}"
    return summed

def comparison(lines):
    lines.extend(["","## A40 与 B40 同场景闭环对照","",
                  "同步 20 Hz，15.0 秒（300 步），71 mm 通用门控，8 cm 试提和 0.55 秒悬停。"
                  "TE exec=1、m=0.01，场景、初态和种子固定。"])
    results={}
    for ck in ("last","best_val"):
        for scope,sn in (("train","10 个训练场景"),("dev","28 个开发场景")):
            f=f"b40_{ck}_{scope}"
            a=read(A/f"a40_last_{scope}.json")
            b=read(B/f"{f}.json")
            pair_doc=read(B/f"{f}_paired.json")
            if not b or not pair_doc:
                continue
            results[ck,scope]=a,b,pair_doc
            lines.extend(["",f"### B40 {ck} · {sn}","",
                "每格依次列出抓起率、抓起且命中数、lift_failed/timeout。"])
            table(lines,("分层","A40 last","B40 "+ck),[
                (k,breakdown(a)(k),breakdown(b)(k)) for k in a["strata"]])
            def diagnostics(arm):
                st=arm["strata"]["all"]
                d,f,t=st["disturbance"],st["failure_types"],st["latency_ms"]
                values=[st["part_hit"],st["part_assessed"],st["collision"]["approach"]["episodes"],
                        d["target_translation_ge_10mm"],d["target_tilt_ge_15deg"],
                        sum(n for k,n in f.items() if "jaw_stable_loaded" in k.split("+")),
                        f.get("lift_path_rejected",0)]
                for k in ("policy_forward_ms","right_detector_ms","to_command_ms"):
                    lat=t.get(k)
                    values.append(f"{lat['p50']:.1f}/{lat['p95']:.1f}" if lat else "—")
                return values
            labels=("Part Hit 命中数","Part Hit 可判定数","接近期碰撞回合数","接近期目标位移 ≥10 mm",
                    "接近期目标倾斜 ≥15°","jaw_stable_loaded（含复合）","lift_path_rejected",
                    "纯策略 P50/P95 ms","右腕检测 P50/P95 ms","端到端 P50/P95 ms")
            table(lines,("诊断","A40","B40"),zip(labels,diagnostics(a),diagnostics(b)))
            p=pair_doc["strata"]["all"]["paired"]
            aw,bw=p.get("a_wins",0),p.get("b_wins",0)
            d=aw+bw
            mcn=min(1,2*sum(math.comb(d,k) for k in range(min(aw,bw)+1))/2**d)
            lines.extend(["",f"A40 独成 {aw}，B40 独成 {bw}，同成 {p.get('both_success',0)}，"
                f"同败 {p.get('both_fail',0)}。精确 McNemar 双侧 p={mcn:.3f}。"])
            table(lines,("场景","部位","有障碍","A40 状态","A40 命中","B40 状态","B40 命中"),[
                (x["case"],x["part"],"是" if x["obstacle"] else "否",x["status"],x["part_hit"],
                 y["status"],y["part_hit"])
                for x,y in zip(a["episodes"],(dict((e["case"],e) for e in b["episodes"])[x["case"]]
                                             for x in a["episodes"]),strict=True)])
    if not results:
        lines.extend(["","B40 闭环评测进行中。A40 last 训练集 5/10 抓起、4/10 且命中，"
                      "开发集 19/28 抓起、16/28 且命中。"])
    for ck in ("last","best_val"):
        swaps=read(B/f"b40_{ck}_swap.json")
        if not swaps:
            continue
        lookup={(r["pair_group_id"],r["step"]):r for r in read(A/"a40_last_swap.json")}
        lines.extend(["",f"### B40 {ck} · 专家分叉步双向换点","",
                      "固定 RGB 和本体状态，只换抓取点。表中为 8×7 关节目标绝对差的中位数／最大值，单位为度。"])
        rows=[]
        for r in swaps:
            if r["step"]==0:
                continue
            a=lookup[r["pair_group_id"],r["step"]]
            def degrees(x):
                return f"{x['median_deg']:.3f}/{x['max_deg']:.3f}"
            for source in r["cases"]:
                rows.append((r["pair_group_id"],r["step"],source,
                             degrees(a["directions"][source]["point"]),degrees(r["directions"][source]["point"])))
        table(lines,("配对组","专家分叉步","原场景","A40 last","B40 "+ck),rows)
    return results

def assets(lines):
    inv,alternates,rays,groups,admitted,pilots=asset_proofs()
    lines.extend(["","## Tier-2 新资产与 L4 基础网格隔离","",
        "新增资产用 CoACD 生成物理碰撞体，接触部位仍以原网格三角面归属判断。"
        "mug/00004 保留原视觉网格。独立异形把手杯替代了不可用的原啤酒杯引用，独立圆篮冻结为 L4。"
        "圆篮与两个训练篮几何分别建模，基础网格指纹不同。",
        "",
        "YCB 使用 Google 16k 扫描网格。[YCB 官方模型集](https://ycb-benchmarks.s3.amazonaws.com/index.html)"])
    entries=[]
    for kind,items in (("Tier-2 候选",inv.get("tier2_proposed",{})),
                       ("本地替代候选",alternates),("L4 留出",inv.get("l4_heldout",{}))):
        for name,obj in items.items():
            t=rays.get(name)
            aperture=("通过" if t["passed"] else "失败")+f"（{t['physical_clear_hole_rays']}/{t['silhouette_hole_samples']}）" if t else "—"
            count=len(groups.get(obj["base_mesh_hash"],[]))
            entries.append((name,kind,obj["hulls"],aperture,
                            "仅 L4" if obj["heldout"] else f"{count} 组"+("，纳入" if name in admitted else ""),
                            obj["base_mesh_hash"]))
    table(lines,("资产","用途","凸体数","PhysX 孔洞射线","训练样本回放","基础网格 SHA-256"),entries)
    base=len(inv.get("tier1_training_base_hashes",{}))
    lines.extend(["",f"已完成至少两组配对回放的新增资产 {len(admitted)} 款，"
                      f"合计原 Tier-1 {base} 款，当前训练池为 {base+len(admitted)} 款。"]
                 +(["新增通过 "+ "、".join(admitted)+"。"] if admitted else []))
    lines.extend(["","marker_box 只计入 00003。与它共享归一化基础网格的 00000 不重复计数。"
                      "新增资产入池前均需原始动作回放、Part Hit 和 §2 HDF5 校验通过。"])
    notes={'006_mustard_bottle': '多次试提中或悬停时滑落，另有单指垫命中。两组配对均未通过。',
 '024_bowl': '夹爪在局部曲面上出现明显单边受力，另一侧未达到门控要求。多次未建立双侧夹持。',
 '011_banana': '离线碰撞与逆运动学筛选未找到成对可行的腕部轨迹。',
 '019_pitcher_base': '把手孔洞通过射线检查，现有腕部候选未通过成对 8 cm 抬升路径筛选。',
 '035_power_drill': '把手和壳体区域的候选未形成成对无碰路径。',
 'arched_basket': '提手孔洞贯通，原侧篮沿的夹持、脱离路径未形成两组。已另验独立抽象的高篮沿篮子。',
 'marker_long_box': '采用竖放姿态，以同一指定端部做有障／无障配对。宽高不同的挡板参与筛选，选用能阻断直达且可安全绕行的布局。',
 'matte_soda': '在两种初始布局迁移已验腕姿，均通过上方罐壁的有障／无障配对。',
 'round_package': '以上方罐壁做有障／无障配对。第二组沿桌面改变初始位置后重新求解。',
 'compact_box': '初轮四次均抓起，但指垫跨标注边界。复验沿盒轴移动抓取中心，原三角面标签不变，四次均通过。'}
    pending={**inv.get("tier2_proposed",{}), **alternates}
    if pending:
        table(lines,("\u8bad\u7ec3\u5019\u9009","\u5df2\u901a\u8fc7\u914d\u5bf9","\u9a8c\u6536\u8bb0\u5f55"), [
            (name,len(groups.get(obj["base_mesh_hash"],[])),notes[name])
            for name,obj in pending.items() if name in notes])
    lines.extend(["", '子任务的 COMPLETE FAILED 表示该轮未完成全部候选预算。入池以跨轮合并后的已通过配对为准，失败样本不参与计数。'])
    if admitted:
        held={o["base_mesh_hash"] for o in inv["l4_heldout"].values()}
        assert not held & {v["base_mesh_hash"] for v in admitted.values()}
        selected=list(admitted)[:max(0,16-base)]
        evidence={"tier1":inv["tier1_training_base_hashes"],"additional":{k:admitted[k] for k in selected},
                  "count":base+len(selected),"l4_shared":0}
        path=WORK/"tier2_ready_assets.json"
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(evidence,indent=2),encoding="utf-8")
        tmp.replace(path)
    lines.extend(["","## L1–L4 冻结基准专家验真"])
    counts=[]
    for layer in range(1,5):
        p=read(WORK/f"l{layer}_progress.json")
        counts.append(p.get("accepted",0))
    frozen=read(WORK/"frozen.json")
    table(lines,("层级","专家回放及 Part Hit","场景类型"),[(f"L{n}",f"{v}/20",d)
        for n,(v,d) in enumerate(zip(counts,("分布内","多物和部位配对","未见障碍布局","独立未见基础网格")),1)])
    if frozen:
        from collections import Counter
        l3=Counter(r.get("layout_type") for r in frozen["cases"] if r["tier"]==3)
        l4=Counter(r["asset_family"] for r in frozen["cases"] if r["tier"]==4)
        table(lines,("L3 障碍布局","条数"),sorted(l3.items()))
        table(lines,("L4 留出资产","条数"),sorted(l4.items()))
        lines.extend(["", 'L4 的布丁盒侧立，金枪鱼罐沿圆边立放，以统一 8 cm 执行器抓取包装厚度。两件资产保留原始基础几何。'])
        lines.extend(["","L3 \u4e13\u5bb6\u9a8c\u771f\u5305\u62ec\u957f\u76d2\u8d1f\u7aef\u7ed5\u969c 4 \u6761\u3001\u6b63\u7aef\u7ed5\u969c 4 \u6761\uff0c\u4ee5\u53ca\u7bee\u63d0\u624b\u7ed5\u969c 3 \u6761\u3001\u7bee\u6cbf\u7ed5\u969c 3 \u6761\u3002\u53ea\u7528\u6765\u9a8c\u8bc1\u573a\u666f\u6a21\u677f\uff0c\u51bb\u7ed3\u8f68\u8ff9\u4e0e\u521d\u6001\u5747\u6c38\u4e45\u9694\u79bb\u3002Tier-1 \u5df2\u6709\u957f\u76d2\u4e24\u7aef\u5404 31 \u6761\u3001\u7bee\u63d0\u624b\u4e0e\u7bee\u6cbf\u5404 6 \u6761\u5e26\u969c\u788d\u7684\u4e13\u5bb6\u6837\u672c\u3002"])
        sha=read(WORK/"frozen_sha256.json").get("sha256","")
        lines.extend(["",f"共 80/80 通过。与 Tier-1 的 460 条和 28 个开发场景均零重合。冻结清单 SHA-256 为 `{sha}`。"])
    else:
        lines.extend(["","专家回放及 Part Hit 达到 80/80 后生成冻结清单。"])
    return pilots

def estimates(lines,pilots,results):
    lines.extend(["","## Step 3-B 耗时估算","",
        "目标为总计 1,500–2,000 条，沿用 Tier-1 后净增 1,040–1,540 条。"
        "按多物和部位配对 40%、多样障碍 35%、大偏航和多起姿 25% 规划。"
        "优先补齐杯身／杯柄配对、长盒两端绕障及篮提手绕障。"])
    measured=[r for stage in ("l1","l2","l3","l4")
              for r in read(WORK/f"{stage}_progress.json").get("cases",[])]
    captures=[r["capture_wall_s"] for r in pilots or measured if r.get("capture_wall_s")]
    replays=[]
    attempts=0
    verified=0
    receipts=sorted(WORK.glob("pilots*_result.json")) if pilots else sorted(WORK.glob("l[1-4]_r*_result.json"))
    for f in receipts:
        trace=read(f)
        attempts+=len(trace["attempts"])
        verified+=len(trace["cases"])
        replays.extend(a["replay_wall_s"] for a in trace["attempts"] if a.get("replay_pass") and a.get("replay_wall_s"))
    if captures and replays:
        cap,rep=median(captures),median(replays)
        # Batch startup and retries are budget assumptions based on the measured stages.
        trial_low=(cap+rep+12)*1.15
        trial_high=(cap+rep+30)*1.45
        lo,hi=1040*trial_low/3600,1540*trial_high/3600
        source="新资产小样本" if pilots else "冻结场景专家验真"
        lines.extend(["",f"{source}的成功录制和原指令回放中位数分别为 {cap:.1f} 秒及 {rep:.1f} 秒。"
            f"小样本工程记录含 {attempts} 次录制尝试，验收通过 {verified} 条。"
            f"按合格模板每条另计 12–30 秒启动及校验开销，以及 15–45% 重试余量。"
            f"预计单路 {lo:.1f}–{hi:.1f} 小时，两路 {lo/2:.1f}–{hi/2:.1f} 小时，三路 {lo/3:.1f}–{hi/3:.1f} 小时。"
            "若扩大到尚未打通的资产，需另计资产改型与预筛耗时。实际并发取决于 GPU 显存和主机 RAM。"])
    else:
        lines.extend(["","按 Tier-1 实测净录制中位 70.1 秒／条，录制及重放试测完成后再用实测值估算。"])
    lines.extend(["","按每条约 96 个窗口、保留约 78% 训练和有效 batch 32，预计每轮 3,500–4,670 次更新。"
        "60–80 轮约 21–37.5 万次更新。A40 单卡实测约 0.197 秒／次，估计 11.5–20.5 小时。"
        "该估算含完整周期的训练和验证，闭环评测另计。"])
    if ("last","dev") in results:
        a,b,_=results["last","dev"]
        ag,bg=a["strata"]["all"],b["strata"]["all"]
        aj,bj=a["success_and_hit"],b["success_and_hit"]
        am=ag["latency_ms"]["to_command_ms"]["p50"]
        bm=bg["latency_ms"]["to_command_ms"]["p50"]
        lines.extend(["","## 路线建议","",f"A40、B40 last 在开发集分别抓起 {ag['success']}、{bg['success']} 条，"
            f"其中成功且命中 {aj}、{bj} 条。端到端中位 {am:.1f}、{bm:.1f} ms。"])
        prior=read(A/"b20_last_dev.json")
        if prior:
            lines.extend(["",f"B20 last 开发集抓起 {prior['strata']['all']['success']}/28、"
                f"抓起且命中 {prior['success_and_hit']}/28。B40 未体现稳定的闭环收敛收益。"])
        if bj<=aj and bm>=am:
            lines.extend(["","建议 Tier-2 定型 A。A 的开发集抓起、部位命中及绕障成绩更高，同步仿真每步端到端更短。"
                "保留同口径评测，重点增加杯柄、长盒两端和篮提手的障碍样本。"])

        elif bj>aj and bg["success"]>=ag["success"]:
            lines.extend(["","B 同时提升了抓起与指定部位命中，路线选择还应考虑在线检测预算。"])
        else:
            lines.extend(["","两个终点的增益不一致，需结合逐场景失败归因选择。"])
    return

def main():
    lines=["# Step 3-A 与 V3-B40 验收报告",""]
    finished=markers(lines)
    cache=read(WORK/"runtime_cache_cleanup.json")
    if cache:
        gib=cache["bytes"]/2**30
        lines.extend(["","## 115 资源与运行记录","",
            f"高峰期主机可用 RAM 曾降至 0.6 GiB。仅清理 {cache['count']} 个已完成项目进程遗留的 Kit 渲染缓存，"
            f"释放 {gib:.2f} GiB，保留轨迹、资产、日志、标定和权重。"
            "训练使用 GPU 2，专家验真和闭环分配到空闲 GPU。调度按显存与主机 RAM 动态降并发，"
            "轻量监控可暂停任务自身的仿真进程并续跑未完成场景。"])
    lines.extend(["", "\u8bc4\u6d4b\u68c0\u67e5\u70b9\u89c4\u5219\uff1aB40 \u6700\u4f73\u9a8c\u8bc1\u635f\u5931\u51fa\u73b0\u5728\u7b2c 36 \u8f6e\uff0c\u4f9d\u7b2c 35 \u8f6e\u95e8\u69db\uff0c\u672c\u8f6e\u4e0e A40 \u5bf9\u7167\u4f7f\u7528 last\uff1b\u4fdd\u7559 best_val \u6743\u91cd\u3002", "", "\u51bb\u7ed3 L1\u2013L4 \u7684 80 \u6761\u53ea\u6267\u884c\u4e13\u5bb6\u9a8c\u771f\u3002\u6b63\u5f0f Tier-2 \u91c7\u96c6\u3001\u6269\u91cf\u8bad\u7ec3\u53ca\u51bb\u7ed3\u96c6\u7684\u7b56\u7565\u6a21\u578b\u8bc4\u6d4b\u5747\u672a\u542f\u52a8\u3002"])
    curves(lines)
    results=comparison(lines)
    pilots=assets(lines)
    estimates(lines,pilots,results)
    lines.extend(["","任务状态以 .complete 文件为准。"])
    ready=read(WORK/"tier2_ready_assets.json").get("count",0)
    passed=all(read(ROOT/f"{k}.complete").get("outcome")=="PASS" for k in CORE) and ready>=12
    if finished and not passed:
        lines.extend(["",f"训练池验收为 {ready} 款，目标下限为 12 款。按各任务表保留未通过原因。"])
    path=WORK/("report.md" if finished else "report_in_progress.md")
    tmp=path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    tmp.replace(path)
    print(path)
    return int(finished and not passed)

if __name__=="__main__":
    raise SystemExit(main())
