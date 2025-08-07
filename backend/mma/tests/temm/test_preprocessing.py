import os
import json
import shutil
import unittest
import numpy as np

from mma_wrapper.temm.io_utils import load_trajectories
from mma_wrapper.temm.preprocessing import (
    extract_full_trajectories,
    extract_action_trajectories,
    extract_observation_trajectories,
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_analysis"
        os.makedirs(os.path.join(self.test_dir, "trajectories"), exist_ok=True)

        # Create fake trajectory data (3D vector obs + int action)
        self.sample_data = {
            "agent_0": [[[1, 2, 3], 0], [[4, 5, 6], 1]],
            "agent_1": [[[7, 8, 9], 2], [[10, 11, 12], 3]]
        }

        for i in range(2):  # Two episodes
            with open(os.path.join(self.test_dir, "trajectories", f"trajectories_{i}.json"), "w") as f:
                json.dump(self.sample_data, f)

        self.raw_episodes = load_trajectories(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extract_full_trajectories(self):
        full_trajectories = extract_full_trajectories(self.raw_episodes)
        self.assertEqual(len(full_trajectories), 4)  # 2 episodes × 2 agents
        for traj in full_trajectories:
            self.assertTrue(all(isinstance(obs, np.ndarray)
                            and isinstance(act, int) for obs, act in traj))

    def test_extract_action_trajectories(self):
        actions = extract_action_trajectories(self.raw_episodes)
        self.assertIn("agent_0", actions)
        self.assertIn("agent_1", actions)
        self.assertEqual(actions["agent_0"], [0, 1, 0, 1])
        self.assertEqual(actions["agent_1"], [2, 3, 2, 3])

    def test_extract_observation_trajectories(self):
        observations = extract_observation_trajectories(self.raw_episodes)
        self.assertIn("agent_0", observations)
        self.assertIn("agent_1", observations)
        self.assertEqual(len(observations["agent_0"]), 4)
        self.assertEqual(len(observations["agent_1"]), 4)
        for obs in observations["agent_0"] + observations["agent_1"]:
            self.assertIsInstance(obs, np.ndarray)


if __name__ == "__main__":
    unittest.main()
