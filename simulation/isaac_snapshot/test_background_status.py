import json
import tempfile
import unittest
from pathlib import Path

from background_status import complete_from_exit, record_complete, run_base, status


class BackgroundStatusTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = run_base(Path(directory.name), "probe_01")

    def test_pending_is_not_complete(self):
        self.base.with_suffix(".exit").write_text("", encoding="utf-8")
        self.assertIsNone(complete_from_exit(self.base))
        self.assertFalse(self.base.with_suffix(".complete").exists())
        self.assertIn("PENDING", status(self.base))

    def test_legacy_failure_is_complete(self):
        self.base.with_suffix(".exit").write_text("9\n", encoding="utf-8")
        marker = complete_from_exit(self.base)
        self.assertEqual(marker["exit_code"], 9)
        self.assertIn("COMPLETE FAILED exit=9", status(self.base))

    def test_success_marker_is_immutable(self):
        first = record_complete(self.base, 0, 1)
        self.assertEqual(record_complete(self.base, 0, 2), first)
        self.assertIn("COMPLETE PASS exit=0", status(self.base))
        with self.assertRaises(ValueError):
            record_complete(self.base, 1)

    def test_planning_result_shows_offline_feasibility(self):
        self.base.mkdir()
        (self.base / "frozen_cases.json").write_text("{}", encoding="utf-8")
        (self.base / "B_042.json").write_text("{}", encoding="utf-8")
        self.assertIn("planned_cases_checked=1", status(self.base))
        result = {"cases": [{"offline_feasible": True}, {"offline_feasible": False}]}
        (self.base / "result.json").write_text(json.dumps(result), encoding="utf-8")
        record_complete(self.base, 0)
        self.assertIn("cases=2 offline_feasible=1", status(self.base))


if __name__ == "__main__":
    unittest.main()
