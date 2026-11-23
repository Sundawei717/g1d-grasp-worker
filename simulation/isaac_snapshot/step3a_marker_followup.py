"""Finish a second marker-package detour group only after the reference passes."""
import time

from background_status import record_complete
from step3a_round_followup import plan_pair
from step3a_suites import CASE_PLANS, RUNS, WORK, read, verify_groups

JOB = "step3a_01_marker_followup"


def main():
    receipt = WORK / "pilots_guide_marker_detour_result.json"
    done = RUNS / "step3a_01_marker_detour.complete"
    while not done.is_file():
        time.sleep(15)
    source = read(receipt).get("cases", [])
    if len(source) >= 4:
        return
    if len(source) != 2 or not all(c["replay_verified"] for c in source):
        raise RuntimeError("Marker-package reference did not verify a complete grasp/detour pair")
    original, obstacle = (read(CASE_PLANS / f"{r['case_id']}.json") for r in source)
    for index, offset in enumerate(((.009, .005), (-.010, .004), (.003, -.012))):
        group = plan_pair(original, obstacle, offset, index, label="marker", seed_start=2471900, first_id=9990)
        print("marker follow-up", index, offset, "planned" if group else "IK/clearance rejected", flush=True)
        if group:
            result = verify_groups([group], f"pilots_guide_marker_followup_r{index}")
            if len(result["cases"]) == 2:
                return
            break
    raise RuntimeError("Second marker-package group did not pass screening and native replay")


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    finally:
        record_complete(RUNS / JOB, code)
