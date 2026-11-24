import argparse
import json
from pathlib import Path

from audit_grasp_receipt import audit


def review(source):
    batch = json.loads((source / "result.json").read_text())
    if batch.get("phase") != "asset_admission_10" or len(batch["cases"]) != 10:
        raise ValueError("Expected a complete physical five-pose asset batch")
    rows = []
    for original in batch["cases"]:
        case = source / original["case"]
        report = json.loads((case / "expert_diagnostic.json").read_text())
        if report["asset_usd"] != batch["asset_usd"] or not report.get("initial_prompt"):
            raise ValueError("Missing case identity or visible part")
        evidence = audit(case, baseline_contract="stable_support_v2")
        rows.append({"case": original["case"], "status": "strict_success" if evidence["strict_success"] else "physical_fail",
                     "original_status": original["status"], "original_score": original["strict_score"], "audit": evidence})
    successes = sum(row["status"] == "strict_success" for row in rows)
    expected = {f"pose_{index:02d}_rep{repeat}" for index in range(5) for repeat in range(2)}
    if {row["case"] for row in rows} != expected:
        raise ValueError("Frozen pose manifest does not match the ten physical receipts")
    return {"phase": "asset_admission_10", "asset_usd": batch["asset_usd"], "baseline_contract": "stable_support_v2",
            "original_batch": str((source / "result.json").resolve()), "source_directory": str(source.resolve()),
            "strict_success": successes, "attempted": 10, "cases": rows, "passed": successes == 10}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evidence = review(args.source)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2)
    print("Independent pre-lift support review:", evidence["strict_success"], "/", evidence["attempted"], flush=True)
    return int(not evidence["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
