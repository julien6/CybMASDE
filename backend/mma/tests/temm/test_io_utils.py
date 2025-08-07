import os
import json
import shutil
import unittest
import numpy as np

from mma_wrapper.temm.io_utils import load_trajectories, export_to_json

class TestIOUtils(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_analysis"
        os.makedirs(os.path.join(self.test_dir, "trajectories"), exist_ok=True)

        self.sample_data = {
            "agent_0": [[[1, 2, 3], 0], [[4, 5, 6], 1]],
            "agent_1": [[[7, 8, 9], 2], [[10, 11, 12], 3]]
        }

        for i in range(3):
            with open(os.path.join(self.test_dir, "trajectories", f"trajectories_{i}.json"), "w") as f:
                json.dump(self.sample_data, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_trajectories(self):
        trajectories = load_trajectories(self.test_dir)
        self.assertEqual(len(trajectories), 3)
        for traj in trajectories:
            self.assertIn("agent_0", traj)
            self.assertIn("agent_1", traj)

    def test_export_to_json(self):
        test_output = os.path.join(self.test_dir, "test_output.json")
        test_dict = {"a": 1, "b": [1, 2, 3]}
        export_to_json(test_dict, test_output)

        self.assertTrue(os.path.exists(test_output))

        with open(test_output, "r") as f:
            loaded = json.load(f)
            self.assertEqual(loaded, test_dict)


if __name__ == "__main__":
    unittest.main()
