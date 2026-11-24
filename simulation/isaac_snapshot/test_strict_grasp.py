import unittest
from dataclasses import replace

from strict_grasp import PhysicsSample, evaluate


class StrictGraspTest(unittest.TestCase):
    def setUp(self):
        pads = frozenset(("pad_1", "pad_2"))
        self.samples = [
            PhysicsSample(index / 100, 0.8 if index < 10 else 0.825, 0.9 if index < 10 else 0.93,
                          (0, 0, 0), (0, 0, 0, 1), index < 10, pads, False, False)
            for index in range(61)
        ]

    def score(self):
        return evaluate(self.samples, lift_start=0, hold_start=10, dt=0.01)

    def test_valid_trace(self):
        self.assertTrue(self.score()["success"])

    def test_wrong_part_collision_and_slippage(self):
        self.samples[10] = replace(self.samples[10], target_part_pads=frozenset(("pad_1",)))
        self.samples[6] = replace(self.samples[6], forbidden_contact=True)
        self.samples[40] = replace(self.samples[40], relative_xyz_m=(0.02, 0, 0))
        self.assertEqual(set(self.score()["reasons"]), {
            "wrong_target_or_part_contact", "forbidden_contact", "target_slipped"})

    def test_contact_with_other_support_or_pose_override(self):
        self.samples[-1] = replace(self.samples[-1], target_supported=True, pose_override_or_constraint=True)
        self.assertEqual(set(self.score()["reasons"]), {"target_not_lifted_clear", "nonphysical_grasp"})

    def test_missing_or_short_hold_samples_fail_closed(self):
        self.samples.pop()
        self.assertIn("short_hold", self.score()["reasons"])
        self.samples.pop(30)
        self.assertEqual(self.score()["reasons"], ["invalid_or_gapped_physics_trace"])

    def test_recent_stable_support_is_distinct_from_a_pre_lift_jump(self):
        self.samples[1] = replace(self.samples[1], time_s=0.01, target_supported=False, target_z_m=0.80008)
        observed = evaluate(self.samples, lift_start=1, hold_start=10, dt=0.01, baseline_contract="stable_support_v2")
        self.assertTrue(observed["success"])
        self.samples[1] = replace(self.samples[1], target_z_m=0.803)
        observed = evaluate(self.samples, lift_start=1, hold_start=10, dt=0.01, baseline_contract="stable_support_v2")
        self.assertIn("invalid_pre_lift_baseline", observed["reasons"])


if __name__ == "__main__":
    unittest.main()
