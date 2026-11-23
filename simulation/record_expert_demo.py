"""Record and replay two expert demonstrations in an installed G1D simulation project."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys


MUG_CASES = {"C_9810": "mug_body", "C_9811": "mug_handle"}
DEVELOPMENT = Path("runs/step2_v3_01/development.json")
REQUIRED_FILES = (
    DEVELOPMENT,
    Path("env_resolved.json"),
    Path("run_lab_school.sh"),
    Path("step3a_verify.py"),
    Path("runs/lab_school_03/result.json"),
    Path("runs/three_view_step0_01/plans/expert_carton.json"),
)


def load_pair(project):
    missing = [str(project / path) for path in REQUIRED_FILES if not (project / path).is_file()]
    if missing:
        raise ValueError("Missing installed project files:\n" + "\n".join(missing))

    cases = json.loads((project / DEVELOPMENT).read_text(encoding="utf-8"))["cases"]
    pair = [row for row in cases if row["case_id"] in MUG_CASES]
    parts = {row["case_id"]: row["part"] for row in pair}
    groups = {row["pair_group_id"] for row in pair}
    if len(pair) != 2 or parts != MUG_CASES or len(groups) != 1 or not next(iter(groups)):
        raise ValueError("The development manifest must contain the C_9810/C_9811 mug pair")
    for row in pair:
        plan = project / row["plan"]
        if not plan.is_file():
            raise ValueError(f"Missing expert plan: {plan}")
        row["plan"] = str(plan)
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True, help="Installed Linux G1D simulation project")
    parser.add_argument("--check", action="store_true", help="Check project files and the two plans without starting Isaac Sim")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    try:
        pair = load_pair(project)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{error}\nUse the configured simulation project; see simulation/README.md.\n")

    parts = ", ".join(f"{row['case_id']} ({row['part']})" for row in pair)
    print(f"Expert plans found: {parts}", flush=True)
    if args.check:
        return 0

    # Import the verifier and configuration from the installed simulator project.
    os.chdir(project)
    sys.path.insert(0, str(project))
    from step3a_verify import WORK, verify_groups

    tag = datetime.now().strftime("handoff_mug_%Y%m%d_%H%M%S_%f")
    print(f"Recording and replaying. Result: {WORK / (tag + '_result.json')}", flush=True)
    result = verify_groups([pair], tag)
    for row in result["cases"]:
        print(f"{row['part']}: {row['expert_path']}")
    if len(result["cases"]) == 2:
        print("PASS: 2/2 demonstrations recorded, replayed, and matched the requested part.")
        return 0

    for attempt in result["attempts"]:
        diagnostic = attempt.get("replay_diagnostic") or attempt.get("record_diagnostic")
        print(f"{attempt['case']}: {diagnostic}")
    print("FAIL: The full pair did not pass. Check the result JSON and diagnostics.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
