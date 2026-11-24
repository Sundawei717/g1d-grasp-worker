import tempfile
import unittest

import h5py
import numpy as np

from dataset_contract import action_windows, policy_inputs, validate


class DatasetContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.file = h5py.File(f"{self.temp.name}/episode.h5", "w")
        self.addCleanup(self.file.close)
        h5 = self.file
        h5["meta/joint_names"] = ["shoulder", "elbow"]
        h5["meta"].attrs.update({"camera_path": "/World/head/Camera", "arm_side": "right", "joint_unit": "rad", "width_unit": "m"})
        h5["obs/timestamp"] = np.arange(8) / 20
        h5["obs/frame_id"] = [f"left:{index}" for index in range(8)]
        h5["obs/rgb_left"] = np.zeros((8, 16, 24, 3), dtype=np.uint8)
        h5["obs/qpos"] = np.zeros((8, 2))
        h5["obs/gripper_width"] = np.full(8, 0.05)
        h5["action/qpos_target"] = np.arange(14).reshape(7, 2) / 100
        h5["action/gripper_width_target"] = np.full(7, 0.05)
        h5["action/native_command"] = np.zeros((7, 4))
        h5["phase"] = ["pre_grasp", "approach", "close", "close_hold", "lift", "hold", "hold"]
        h5["train_action_valid"] = np.arange(7) < 4
        h5["policy_prompt/bbox"] = np.tile([0.1, 0.25, 0.4, 0.55], (8, 1))
        h5["policy_prompt/point"] = np.tile([0.25, 0.42], (8, 1))
        h5["policy_prompt/valid"] = np.ones(8, dtype=bool)

    def validate(self):
        return validate(self.file, joint_names=["shoulder", "elbow"], camera_path="/World/head/Camera",
                        resolution=(24, 16), q_min=(-1, -1), q_max=(1, 1), width_limits=(0.001, 0.091))

    def test_train_windows_begin_at_zero_and_stop_at_lift(self):
        self.assertEqual(self.validate().valid_training_actions, 4)
        results = list(action_windows(self.file["action/qpos_target"], self.file["train_action_valid"]))
        self.assertEqual([start for start, _, _ in results], [0, 1, 2, 3])
        for start, actions, mask in results:
            np.testing.assert_array_equal(actions[mask], self.file["action/qpos_target"][start:4])
            self.assertTrue(np.all(actions[~mask] == 0))

    def test_truth_group_is_not_needed(self):
        self.assertNotIn("gt", self.file)
        observations = policy_inputs(self.file, 0, min_crop_px=8)
        self.assertEqual(observations["recent_state"].shape, (2, 3))
        self.assertEqual(observations["global_view"].shape, (5, 224, 224))

    def test_stale_time_and_lift_leakage_are_rejected(self):
        self.file["obs/timestamp"][3] = 0
        with self.assertRaisesRegex(ValueError, "clock"):
            self.validate()
        self.file["obs/timestamp"][3] = 0.15
        self.file["train_action_valid"][4] = True
        with self.assertRaisesRegex(ValueError, "trial-lift"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
