"""Render the completed Step 2 comparison."""
from collections import Counter
from math import comb
from pathlib import Path

from background_status import record_complete
from step2_dev import RUNS, WORK, groups
from step2_eval import wait
from three_view_eval import read


LABELS = {"all": "??", "mug": "???", "basket": "????", "carton": "??",
          "mug_body": "??", "mug_handle": "??", "basket_handle": "????", "basket_rim": "????",
          "box_negative_end": "????", "box_positive_end": "????",
          "obstacle": "??", "non_obstacle": "?????"}
METRICS = ("policy_forward_ms", "right_detector_ms", "to_command_ms")
VERDICTS = ("a_wins", "b_wins", "both_success", "both_fail")
WINS = dict(zip(VERDICTS, ("A ?", "B ?", "??", "??")))


def comparison(scope):
    data = read(WORK / f"{scope}_comparison.json")
    strata = data["strata"]
    title = "10 ?????????" if scope == "train" else "28 ???????????"
    out = ["", f"## {title}", "",
           "| ?? | ??? | A ??? | A ???? / ?? | B ??? | B ???? / ?? | Part Hit A / B | A ? / B ? / ?? / ?? |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key,d in strata.items():
        a,b = d["a"],d["b"]
        n = a["count"]
        def success(row):
            return f"{row['success']}/{n}?{row['success']/n:.1%}?"
        def failures(row):
            return f"{row['lift_failed']} / {row['timeout']}"
        paired = " / ".join(str(d["paired"].get(k,0)) for k in VERDICTS)
        out.append(f"| {LABELS[key]} | {n} | {success(a)} | {failures(a)} | {success(b)} | {failures(b)} | "
                   f"{a['part_hit']}/{n} / {b['part_hit']}/{n} | {paired} |")
    out.extend(["","| ?? | ?????? / ????????? | ?????? / ????????? | ???? ?10 mm / ?? ?15? | ???? ?10 mm / ?? ?15? |",
                "| --- | ---: | ---: | ---: | ---: |"])
    for v in "ab":
        d = strata["all"][v]
        ap,cl = (d["collision"][k] for k in ("approach","close"))
        m = d["disturbance"]
        out.append(f"| {v.upper()} | {ap['episodes_with_contact']} / {ap['arm_link_onsets']} | "
                   f"{cl['episodes_with_contact']} / {cl['arm_link_onsets']} | "
                   f"{m['target_translation_ge_10mm']} / {m['target_tilt_ge_15deg']} | "
                   f"{m['fixture_translation_ge_10mm']} / {m['fixture_tilt_ge_15deg']} |")
    out.extend(["","| ?? | ??? P50 / P95 ms | ???? P50 / P95 ms | ??? P50 / P95 ms |",
                "| --- | ---: | ---: | ---: |"])
    for v in "ab":
        d = strata["all"][v]["latency_ms"]
        bins = [f"{d[k]['p50']:.1f} / {d[k]['p95']:.1f}" if k in d else "?" for k in METRICS]
        out.append(f"| {v.upper()} | " + " | ".join(bins) + " |")
    out.extend(["", "???????????????????????", ""])
    for v in "ab":
        d = strata["all"][v]
        types = "?".join(f"{reason} ? {number}" for reason,number in d["failure_types"].items()) or "?"
        out.append(f"- {v.upper()} ?????? Part Hit ? {d['part_hit']}/{d['part_assessed']}????? {types}?")
    out.extend(["","| ?? | ????? | ?? | A | B | ?? |",
                "| --- | --- | --- | --- | --- | --- |"])
    for p in data["paired"]:
        obstacle = "?" if p["obstacle"] else "?"
        out.append(f"| {p['case']} | {LABELS[p['asset']]} ? {LABELS[p['part']]} | {obstacle} | "
                   f"{p['a']} | {p['b']} | {WINS[p['outcome']]} |")
    out.extend(["","?????????",""])
    for v in "ab":
        seen = set()
        for e in data["episodes"][v]:
            cause = e["failure"]
            if not cause or cause in seen or len(seen) == 3:
                continue
            seen.add(cause)
            g = e["gate_passed_steps"]
            detail = e["failure_detail"] or (f"?????????????????? "
                     f"{g['sustained_close']} / {g['arm_stable']} / {g['jaw_stable_loaded']} ??"
                     f"?????? {g['ready']} ?")
            detail = str(detail).replace("\n"," ")
            out.append(f"- {v.upper()} {e['case']} ? {e['status']}?????? {cause}?{detail}?")
    return out, strata["all"]["paired"]


def report():
    wait(("train_a", "train_b", "dev", "eval_train", "eval_dev", "eval_best"))
    dev = read(WORK / "development.json")
    rounds = [read(path) for path in sorted(WORK.glob("expert_round_*.json"))]
    out = ["# V3-Final Step 2 ??? A/B ????", "",
           "## ?????????????", "",
           f"?? **{dev['count']}/28** ????????? 8 cm ???????? Part Hit?"
           f"?????? {rounds[0]['accepted']}/28????? {len(dev['replacements'])} ??"
           "???? 10?? 10??? 8??? 10 ????", ""]
    failures = Counter()
    for rnd in rounds:
        for a in rnd["attempts"]:
            if a.get("failure"):
                failures[a["failure"]] += 1
            elif not a.get("passed") and Path(a["diagnostic"]).exists():
                diagnostic = read(a["diagnostic"])
                roll = diagnostic.get("policy_rollout", {})
                if roll.get("success") and roll.get("part_hit") is not True:
                    failures["??????????????"] += 1
                elif a.get("replay_status") and a["replay_status"] != "success":
                    failures[f"?? {a['replay_status']}"] += 1
                elif "passed" in a:
                    failures["??????????"] += 1
    out.extend([f"- {name} ? {count}" for name,count in failures.items()])
    alignment = [r["expert_alignment"] for r in dev["cases"]]
    out.extend(["",f"???????? {max(abs(r['trigger_delta_s']) for r in alignment)*1e3:.1f} ms?"
                f"??????? {max(r['max_timestamp_error_s'] for r in alignment)*1e6:.3f} ?s?", "",
                "????????????? 5 ??????????????17 ???????3 ???????"
                "?? 5 ?????????????????????????", "",
                "## ???????", "",
                "?? 359 ???? 101 ?????? seed 33889?2,000 ?????? batch 32?4 ? 8??"
                "AdamW???????????? stem 1e-5??? 1e-4?????? 1e-4????? 1.0?"
                "?? BN ???????? BF16 ???????????????????? 17 ??????", "",
                "| ?? | ??? | ????? | ???? | ???? | ???? | ????? | ???? | ???? | ???? |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    best = {}
    for variant in "ab":
        folder = RUNS / f"{WORK.name}_train_{variant}"
        c = read(folder / "loss_curves.json")
        proof = read(folder / "supervision.json")["phases"]
        if sum(r["close_terms"] for r in c["train"]) != proof["close"]["loss_terms"]:
            raise ValueError(f"Recorded close losses differ from coverage for {variant}")
        t = {r["step"]:r for r in c["train"]}
        for v in c["validation"]:
            step = v["step"]
            cols = [step, *[f"{r[k]:.6f}" for r in (t[step],v) for k in ("total","joint","gripper","close")]]
            out.append(f"| {variant.upper()} | " + " | ".join(map(str,cols)) + " |")
        best[variant] = min(c["validation"], key=lambda v:v["total"])["step"]
    out.extend(["",f"A?B ? best_val ???? {best['a']} ?? {best['b']} ????????? 4,733 ?"
                "??????????? 102,462 ???????? 0?? 500 ?????????????? last ???", "",
                "???????15.0 ?????????? 71 mm ??????8 cm ???0.55 ????"])
    forced = WORK / "teacher_forced10.json"
    if forced.exists():
        out.extend(["", "### ????????????", "",
                    "| ?? | ???? | ????? | ?? MAE??? | ???? MAE?mm? |",
                    "| --- | --- | ---: | ---: | ---: |"])
        for v, data in read(forced).items():
            for phase in ("approach", "close"):
                rows = [r for r in data if r["phase"] == phase]
                count = sum(r["labels"] for r in rows)
                joint, jaw = (sum(r[key]*r["labels"] for r in rows)/count
                              for key in ("joint_mae_deg", "gripper_mae_mm"))
                out.append(f"| {v.upper()} | {phase} | {count} | {joint:.3f} | {jaw:.3f} |")
        out.extend(["", "??????????????????????????????????????"])
    out.extend(comparison("train")[0])
    drift = WORK / "train_closed_loop_drift.json"
    if drift.exists():
        vals = read(drift)
        out.extend(["", "### ??????????", "",
                    "| ?? | ? 0 / 20 / 40 / 60 ???????? | ??????????? |",
                    "| --- | --- | ---: |"])
        for v,d in vals.items():
            differences = " / ".join(f"{x:.2f}?" for x in d["median_joint_deviation_deg"])
            out.append(f"| {v.upper()} | {differences} | {d['arm_slew_clip_steps_median']:.0f} |")
        out.extend(["", "??????????? 7 ?????????????????"
                    "??????????????????????"])

    supplementary = read(WORK / "best_comparison.json")
    best_cases = {p["case"] for p in supplementary["paired"]}
    previous = read(WORK / "train_comparison.json")["paired"]
    out.extend(["", "### best_val ??", "",
                "? 1,500 ??????????? 2,000 ??????????"
                "???????????????????????????????? 1,500 ????", "",
                "| ?? | last ?? / ???? / ?? | best_val ?? / ???? / ?? |",
                "| --- | ---: | ---: |"])
    for v in "ab":
        terminal = ("success","lift_failed","timeout")
        last = Counter(p[v] for p in previous if p["case"] in best_cases)
        earlier = supplementary["strata"]["all"][v]
        out.append(f"| {v.upper()} | " + " / ".join(str(last[t]) for t in terminal) + " | "
                   + " / ".join(str(earlier[t]) for t in terminal) + " |")
    lines, verdicts = comparison("dev")
    out.extend(lines)
    detail = read(WORK / "dev_comparison.json")["episodes"]
    by_case = {v: {e["case"]: e for e in detail[v]} for v in "ab"}
    pairs = groups(dev["cases"])
    out.extend(["", "??????????????", "",
                "| ?? | ??? | A ?????????? | B ?????????? |",
                "| --- | ---: | ---: | ---: |"])
    for asset in ("mug", "basket", "carton"):
        ps = [pair for pair in pairs if pair[0]["bucket"] == asset]
        def achieved(variant, pair):
            es = [by_case[variant][r["case_id"]] for r in pair]
            return all(e["status"] == "success" and e["part_hit"] is True for e in es)
        hits = [sum(achieved(v,pair) for pair in ps) for v in "ab"]
        out.append(f"| {LABELS[asset]} | {len(ps)} | {hits[0]} | {hits[1]} |")
    a,b = (verdicts.get(k,0) for k in VERDICTS[:2])
    n = a+b
    p = min(1,2*sum(comb(n,k) for k in range(min(a,b)+1))/2**n)
    if not a and not b and not verdicts.get("both_success"):
        choice = "???????????A ?????????????????????????? A ? B ???"
    elif b>a and p<=.05:
        choice = "??????????? B????????????????"
    elif b>a:
        choice = "B ????????????????????? A?B ???????"
    else:
        choice = "?? A????? B ???????????????"
    out.extend(["","## ????","",f"??? A ?? {a}?B ?? {b}????????? p={p:.3f}?{choice}","",
                "Step 2 ?????Tier-2 ????? 80 ????????",""])
    (WORK / "Step2_V3_AB_Report.md").write_text("\n".join(out),encoding="utf-8")


if __name__ == "__main__":
    base = RUNS / f"{WORK.name}_report"
    try:
        report()
    except BaseException:
        record_complete(base,1)
        raise
    else:
        record_complete(base,0)
