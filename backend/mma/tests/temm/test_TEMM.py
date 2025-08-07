# test_TEMM.py

import traceback
import unittest
import os
import json
import shutil
import numpy as np

from mma_wrapper.temm.TEMM import TEMM


class TestTEMMIntegration(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory with synthetic trajectory data
        self.temp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "temp_test_results")

        os.makedirs(self.temp_path, exist_ok=True)

        # Init TEMM instance
        self.temm = TEMM(self.temp_path)

        # Create mock trajectories for two agents and two episodes
        for i in range(2):
            data = {
                "agent_0": [
                    ([1, 0, 0], 0),
                    ([0, 1, 0], 3)
                ],
                "agent_1": [
                    ([0, 0, 1], 1),
                    ([1, 0, 0], 2)
                ]
            }
            with open(os.path.join(self.temp_path, "trajectories", f"trajectories_{i}.json"), "w") as f:
                json.dump(data, f)

    def test_run_global(self):
        try:
            self.temm.run_global(
                distance_method_action="euclidean",
                distance_method_obs="euclidean",
                distance_method_full="euclidean",
                centroid_mode="mean",
                inclusion_radius=1.0,
                min_rule_weight=0.0,
                min_goal_weight=0.0
            )
        except Exception as e:
            self.fail(
                f"run_global() raised an unexpected exception: {e} -> {traceback.format_exc()}")

        # Check that output JSONs are generated
        roles_file = os.path.join(
            self.temp_path, "inferred_organizational_specifications", "inferred_roles_summary.json")
        goals_file = os.path.join(
            self.temp_path, "inferred_organizational_specifications", "inferred_goals_summary.json")
        self.assertTrue(os.path.exists(roles_file))
        self.assertTrue(os.path.exists(goals_file))

        with open(roles_file, 'r') as f:
            roles = json.load(f)
            self.assertIsInstance(roles, dict)

        with open(goals_file, 'r') as f:
            goals = json.load(f)
            self.assertIsInstance(goals, dict)

    def tearDown(self):
        # Clean up temporary test directory
        shutil.rmtree(self.temp_path)


if __name__ == "__main__":
    unittest.main()
