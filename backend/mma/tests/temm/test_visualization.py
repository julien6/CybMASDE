import unittest
import os
from mma_wrapper.temm.clustering import cluster_full_trajectories
import numpy as np
import shutil

from mma_wrapper.temm.visualization import (
    generate_dendrogram,
    generate_actions_dendrogram,
    generate_observations_dendrogram,
    visualize_action_pca,
    visualize_observation_pca,
    visualize_transition_pca,
    visualize_action_trajectory,
    visualize_observation_trajectory,
    visualize_trajectory,
)


class TestVisualization(unittest.TestCase):

    def setUp(self):

        self.figures_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "figures")
        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path)

        # 3 simple action trajectories (integer values ​​between 0 and 2)
        self.action_trajectories = [
            [0, 1, 2],
            [0, 1, 2],
            [2, 1, 0]
        ]

        # 3 observation trajectories (3-dimensional vectors)
        self.observation_trajectories = [
            [np.array([0, 1, 0]), np.array([0, 1, 1])],
            [np.array([0, 1, 0]), np.array([0, 1, 1])],
            [np.array([1, 0, 1]), np.array([1, 0, 0])]
        ]

        # 3 complete trajectories (obs, act)
        self.full_trajectories = [
            [(np.array([0, 1]), 0), (np.array([1, 0]), 1)],
            [(np.array([0, 1]), 0), (np.array([1, 0]), 1)],
            [(np.array([1, 0]), 2), (np.array([0, 1]), 0)]
        ]

    def tearDown(self):
        if os.path.exists(self.figures_path):
            shutil.rmtree(self.figures_path)

    def test_generate_dendrogram(self):
        output_file_path = generate_dendrogram(self.full_trajectories, os.path.join(self.figures_path, "full_dendrogram.png"))
        self.assertTrue(os.path.exists(output_file_path))

    def test_generate_actions_dendrogram(self):
        output_file_path = generate_actions_dendrogram(self.action_trajectories, os.path.join(self.figures_path, "actions_dendrogram.png"))
        self.assertTrue(os.path.exists(output_file_path))

    def test_generate_observations_dendrogram(self):
        output_file_path = generate_observations_dendrogram(self.observation_trajectories, os.path.join(self.figures_path, "observations_dendrogram.png"))
        self.assertTrue(os.path.exists(output_file_path))

    def test_visualize_action_pca(self):
        output_file_path = visualize_action_pca(self.action_trajectories, save_path=os.path.join(
            self.figures_path, "action_pca.png"))
        self.assertTrue(os.path.exists(output_file_path))

    def test_visualize_observation_pca(self):
        output_file_path = visualize_observation_pca(
            self.observation_trajectories, save_path=os.path.join(self.figures_path, "obs_pca.png"))
        self.assertTrue(os.path.exists(output_file_path))

    def test_visualize_transition_pca(self):
        output_file_path = visualize_transition_pca(
            self.full_trajectories, save_path=os.path.join(self.figures_path, "transition_pca.png"))
        self.assertTrue(os.path.exists(output_file_path))

    def test_visualize_action_trajectory(self):
        output_file_path = visualize_action_trajectory(self.action_trajectories, save_path=os.path.join(
            self.figures_path, "action_trajectory.png"))
        self.assertTrue(os.path.exists(output_file_path))

    def test_visualize_observation_trajectory(self):
        output_file_path = visualize_observation_trajectory(
            self.observation_trajectories, save_path=os.path.join(self.figures_path, "obs_trajectory.png"))
        self.assertTrue(os.path.exists(output_file_path))

    def test_visualize_trajectory(self):
        output_file_path = visualize_trajectory(self.full_trajectories, save_path=os.path.join(
            self.figures_path, "full_trajectory.png"))
        self.assertTrue(os.path.exists(output_file_path))


if __name__ == '__main__':
    unittest.main()
