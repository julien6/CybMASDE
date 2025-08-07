# test_inferring_role.py

import unittest
import numpy as np
from mma_wrapper.temm.inferring_roles import extract_roles_from_trajectories, summarize_roles


class TestInferringRoles(unittest.TestCase):

    def setUp(self):
        # Define mock trajectories per cluster
        # Each trajectory is a list of (observation, action) tuples
        self.mock_clusters = {
            0: [
                [(np.array([1, 0]), 0), (np.array([0, 1]), 1)],
                [(np.array([1, 0]), 0), (np.array([0, 1]), 1)],
            ],
            1: [
                [(np.array([0, 1]), 1), (np.array([1, 1]), 0)],
                [(np.array([0, 1]), 1), (np.array([1, 1]), 0)],
            ]
        }

    def test_extract_roles_from_trajectories(self):
        roles = extract_roles_from_trajectories(self.mock_clusters)
        self.assertIsInstance(roles, dict)
        self.assertIn(0, roles)
        self.assertIn(1, roles)
        self.assertIn("rules", roles[0])
        self.assertGreater(len(roles[0]["rules"]), 0)

    def test_summarize_roles(self):
        roles = extract_roles_from_trajectories(self.mock_clusters)
        summarized = summarize_roles(roles, min_rule_weight=0.5)
        self.assertIsInstance(summarized, dict)
        for cluster_id, role in summarized.items():
            for rule in role["rules"]:
                self.assertGreaterEqual(rule["weight"], 0.5)


if __name__ == "__main__":
    unittest.main()
