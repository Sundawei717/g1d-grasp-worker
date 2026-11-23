"""Record and replay examples in an installed G1D simulation project."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys


TIER1 = Path("runs/tier1_v3_01")
DEVELOPMENT = Path("runs/step2_v3_01/development.json")
TIER1_SPLIT = TIER1 / "training_split.json"
PILOTS = Path("runs/step3a_01")
PRESETS = {
    "mug": (DEVELOPMENT, ("C_9810", "C_9811"), "Brown mug, body vs. handle"),
    "mug-minus75": (DEVELOPMENT, ("C_9812", "C_9813"), "Brown mug at -75 degrees"),
    "mug-plus75-obstacle": (TIER1_SPLIT, ("B_7424", "B_7425"), "Brown mug at +75 degrees with a blocker"),
    "basket": (DEVELOPMENT, ("C_9800", "C_9801"), "Basket, top handle vs. rim"),
    "basket-plus75": (TIER1_SPLIT, ("C_8212", "C_8213"), "Basket at +75 degrees"),
    "basket-obstacle": (DEVELOPMENT, ("B_9832", "B_9833"), "Basket at +34 degrees with a blocker"),
    "long-box": (DEVELOPMENT, ("C_9808", "C_9809"), "Long box at -85 degrees, opposite ends"),
    "long-box-obstacle": (DEVELOPMENT, ("B_9836", "B_9837"), "Long box at +52 degrees with a blocker"),
    "white-box-pepsi": (TIER1_SPLIT, ("A_7004", "A_7132"), "White box vs. Pepsi can in the same layout; rerecords verified archived actions"),
    "sugar-box": (PILOTS / "pilots_sugar_calibration_r1_result.json", ("C_9340", "C_9341"), "YCB sugar box, opposite ends"),
    "ycb-mug": (PILOTS / "pilots_progress.json", ("C_8600", "C_8601"), "YCB mug, body vs. handle"),
    "raised-rim-basket": (PILOTS / "substitutes_progress.json", ("C_9514", "C_9515"), "Raised-rim basket, handle vs. rim"),
    "marker-long-box": (PILOTS / "pilots_guide_marker_followup_r0_result.json", ("C_9990", "B_9991"), "Upright marker box, clear vs. blocked"),
    "compact-box": (PILOTS / "pilots_guide_compact_calibration_r0_result.json", ("C_9940", "B_9941"), "Compact box, clear vs. blocked"),
    "round-package": (PILOTS / "pilots_guide_round_package_r1_result.json", ("C_9850", "B_9851"), "Round package, clear vs. blocked"),
    "matte-soda": (PILOTS / "pilots_guide_matte_soda_retarget_result.json", ("C_9970", "B_9971"), "Matte soda can, clear vs. blocked"),
}
REQUIRED_FILES = (
    Path("env_resolved.json"),
    Path("run_lab_school.sh"),
    Path("step3a_verify.py"),
    Path("runs/lab_school_03/result.json"),
    Path("runs/three_view_step0_01/plans/expert_carton.json"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_pair(project, preset):
    source, case_ids, _ = PRESETS[preset]
    required = (*REQUIRED_FILES, source)
    missing = [str(project / path) for path in required if not (project / path).is_file()]
    if missing:
        raise ValueError("Missing installed project files:\n" + "\n".join(missing))

    manifest = read_json(project / source)
    rows = manifest["episodes" if source == TIER1_SPLIT else "cases"]
    id_field = "case" if source == TIER1_SPLIT else "case_id"
    selected = {row[id_field]: row for row in rows if row[id_field] in case_ids}
    if selected.keys() != set(case_ids):
        raise ValueError(f"The {preset} example needs both recorded cases: {', '.join(case_ids)}")
    groups = {row["pair_group_id"] for row in selected.values()}
    if len(groups) != 1 or not next(iter(groups)):
        raise ValueError(f"The two {preset} cases are no longer in the same verified pair")

    pair = []
    for case_id in case_ids:
        if source == TIER1_SPLIT:
            folder = project / TIER1 / "plans" / case_id
            row = read_json(folder / "ready.json")
            plan = folder / "plan.json"
        else:
            row = selected[case_id]
            plan = project / row["plan"]
        if row["case_id"] != case_id:
            raise ValueError(f"Unexpected scene in {plan}")
        definition = read_json(plan)
        if definition["case"]["case_id"] != case_id or not definition["feasible"]:
            raise ValueError(f"Missing verified feasible plan: {plan}")
        if legacy := row.get("replay_h5"):
            archive = project / legacy["path"]
            if not archive.is_file():
                raise ValueError(f"Missing original actions for rerecording: {archive}")
            row = {**row, "replay_h5": {**legacy, "path": str(archive)}}
        pair.append({**row, "plan": str(plan)})
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, help="Installed Linux G1D simulation project")
    parser.add_argument("--preset", choices=PRESETS, default="mug", help="Which example to record (default: mug)")
    parser.add_argument("--list", action="store_true", help="Show example names and exit")
    parser.add_argument("--check", action="store_true", help="Check the selected pair without starting Isaac Sim")
    args = parser.parse_args()
    if args.list:
        for name, (_, _, label) in PRESETS.items():
            print(f"{name:22} {label}")
        return 0
    if args.project_dir is None:
        parser.error("--project-dir is required to check or record a preset")

    project = args.project_dir.resolve()
    try:
        pair = load_pair(project, args.preset)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{error}\nUse the configured simulation project; see simulation/README.md.\n")

    print(f"Expert plans found for {args.preset}:", flush=True)
    for row in pair:
        yaw = row.get("yaw_deg")
        angle = f"{yaw:+g} deg" if yaw is not None else "archived pose"
        print(f"  {row['case_id']}: {row['asset_family']} / {row['part']} / {angle}", flush=True)
    if args.check:
        return 0

    # Import the verifier and configuration from the installed simulator project.
    os.chdir(project)
    sys.path.insert(0, str(project))
    from step3a_verify import WORK, verify_groups

    tag = datetime.now().strftime("handoff_%Y%m%d_%H%M%S_%f")
    print(f"Recording and replaying. Result: {WORK / (tag + '_result.json')}", flush=True)
    result = verify_groups([pair], tag)
    for row in result["cases"]:
        print(f"{row['case_id']} {row['asset_family']} / {row['part']}: {row['expert_path']}")
    if len(result["cases"]) == len(pair):
        print(f"PASS: {len(pair)}/{len(pair)} demonstrations recorded, replayed, and matched the requested part.")
        return 0

    for attempt in result["attempts"]:
        diagnostic = attempt.get("replay_diagnostic") or attempt.get("record_diagnostic")
        print(f"{attempt['case']}: {diagnostic}")
    print("FAIL: The full pair did not pass. Check the result JSON and diagnostics.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
