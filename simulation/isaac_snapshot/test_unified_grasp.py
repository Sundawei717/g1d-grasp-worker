import unittest
from time import perf_counter
from unittest.mock import patch

from unified_grasp import EPISODE_STEPS, HOLD_STEPS, LIFT_STEPS, LiftGate, LiftGateConfig, execute, guarded_gripper_width


def reading(step, *, width=0.04, displacement=0.0, force=8.0):
    return {"sim_time_s": step / 20, "eef_xyz_m": [displacement, 0, 0], "width_m": width,
            "max_joint_speed_rad_s": 0.2, "pad_normal_force_n": {"pad_1": force, "pad_2": force}}


class UnifiedGraspTests(unittest.TestCase):
    def test_preload_can_accumulate_behind_a_physically_stopped_jaw(self):
        cfg = LiftGateConfig()
        observed = 0.073
        previous, forces = 0.069, {"pad_1": 20, "pad_2": 20}
        self.assertAlmostEqual(guarded_gripper_width(0.051, observed, previous, {}, cfg), 0.069)
        self.assertAlmostEqual(guarded_gripper_width(0.051, observed, previous, forces, cfg), 0.065)

    def test_requires_loaded_convergence_and_a_still_arm(self):
        gate = LiftGate(LiftGateConfig(), reading(0))
        for step in range(1, 6):
            trial = gate.update(0.03, reading(step, displacement=0.02 * step))
        self.assertTrue(trial["sustained_close"])
        self.assertTrue(trial["jaw_stable_loaded"])
        self.assertFalse(trial["arm_stable"])
        for step in range(6, 10):
            trial = gate.update(0.03, reading(step, displacement=0.1, force=0))
        self.assertTrue(trial["arm_stable"])
        self.assertFalse(trial["jaw_stable_loaded"])
        for step in range(10, 13):
            trial = gate.update(0.03, reading(step, displacement=0.1))
        self.assertTrue(trial["ready"])

    def test_empty_close_can_trigger_at_the_mechanical_limit(self):
        gate = LiftGate(LiftGateConfig(), reading(0, width=0.002, force=0))
        for step in range(1, 6):
            state = gate.update(0.0011, reading(step, width=0.002, force=0))
        self.assertTrue(state["ready"])
        self.assertTrue(state["empty_at_mechanical_limit"])

    def test_pad_feedback_uses_physical_contact_without_using_target_identity(self):
        import numpy as np

        from rigid_grasp_probe import ContactObserver

        observer = ContactObserver.__new__(ContactObserver)
        observer.robot_path, observer.target_path, observer.table_path = "/Robot", "/Target", "/Table"
        observer.obstacle_paths, observer.views = ("/Obstacle",), []
        pad_names = [f"right_dex1_finger_link_{i}_tip" for i in (1, 2)]
        planes = dict(zip(pad_names, [(1, 0.020), (-1, -0.020)], strict=True))
        pose = np.array([0, 0, 0, 0, 0, 0, 1])
        for entity in (observer.target_path, observer.obstacle_paths[0]):
            observer.events = []
            for name, (_, y) in planes.items():
                pair = (f"/Robot/{name}", entity)
                observer.events.append((pair, pair, np.array([0.1, y, 0]), 0.0, 7.0))
            observer.sample_contacts("close", pose, pose, planes, (0, 1))
            self.assertEqual(observer.last_tactile_pad_forces_n, {"pad_1": 7.0, "pad_2": 7.0})
            self.assertEqual(len(observer.last_inner_contact_points), 2 if entity == observer.target_path else 0)

    def execute_trial(self, close):
        step = 0

        def frame():
            return None, {"sensor_sim_time_s": step / 20}

        def apply(command):
            nonlocal step
            step += 1
            return command, perf_counter()

        def lift(trigger):
            nonlocal step
            used = LIFT_STEPS + HOLD_STEPS
            step += used
            return {"trigger": trigger, "steps_used": used, "success": False, "part_hit": True}

        cfg = LiftGateConfig()
        return execute(initial_frame=frame(), capture=frame,
                       source=lambda start, *_: ([close(start), close(start + 1)], {}),
                       feedback=lambda: reading(step), apply=apply, lift=lift, config=cfg)

    def test_trial_failure_terminates_without_applying_remaining_policy_action(self):
        result = self.execute_trial(lambda _: 0.03)
        self.assertEqual(result["status"], "lift_failed")
        self.assertEqual(result["trials"][0]["trigger"]["step"], 5)
        self.assertEqual(result["decisions"][-1]["steps_applied"], 1)
        self.assertEqual(result["steps"], 5 + LIFT_STEPS + HOLD_STEPS)

    def test_no_trial_with_insufficient_time_left(self):
        result = self.execute_trial(lambda step: 0.09 if step < 270 else 0.03)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["steps"], EPISODE_STEPS)
        self.assertTrue(result["gate"][-1]["ready"])
        self.assertEqual(result["trials"], [])

    def test_part_hit_uses_nearest_faces_at_the_same_height(self):
        import trimesh

        from part_surface import attribute_contacts

        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 1, 0], [0, 0, 1],
                                         [1, 0, 0], [1, 1, 0], [1, 0, 1]],
                               faces=[[0, 1, 2], [3, 4, 5]], process=False)
        contacts = [{"pad": "pad_1", "target_local_xyz_m": [1.01, 0.1, 0.25]},
                    {"pad": "pad_2", "target_local_xyz_m": [0.99, 0.2, 0.25]}]
        labels = {"body": {"/combined": [0]}, "handle": {"/combined": [1]}}
        with patch("part_surface.load_collision_surfaces", return_value={"/combined": mesh}):
            self.assertTrue(attribute_contacts("unused", contacts, "handle", (0, 1), labels)["hit"])
            self.assertFalse(attribute_contacts("unused", contacts, "body", (0, 1), labels)["hit"])


if __name__ == "__main__":
    unittest.main()
