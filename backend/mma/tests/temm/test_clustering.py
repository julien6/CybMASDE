import unittest
import numpy as np

from mma_wrapper.temm.clustering import (
    cluster_trajectories_from_action,
    cluster_trajectories_from_observation,
    cluster_full_trajectories
)


class TestClustering(unittest.TestCase):

    def setUp(self):
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

    def test_cluster_trajectories_from_action_euclidean(self):
        clusters, linkage_matrix = cluster_trajectories_from_action(
            self.action_trajectories, "euclidean")
        self.assertIsInstance(clusters, dict)
        self.assertGreaterEqual(len(clusters), 1)

    def test_cluster_trajectories_from_action_cosine(self):
        clusters, linkage_matrix = cluster_trajectories_from_action(
            self.action_trajectories, "cosine")
        self.assertIsInstance(clusters, dict)

    def test_cluster_trajectories_from_action_lcs(self):
        clusters, linkage_matrix = cluster_trajectories_from_action(
            self.action_trajectories, "lcs")
        self.assertIsInstance(clusters, dict)

    def test_cluster_trajectories_from_action_fuzzy_lcs(self):
        clusters, linkage_matrix = cluster_trajectories_from_action(
            self.action_trajectories, "fuzzy_lcs")
        self.assertIsInstance(clusters, dict)

    def test_cluster_trajectories_from_action_smith_waterman(self):
        clusters, linkage_matrix = cluster_trajectories_from_action(
            self.action_trajectories, "smith_waterman")
        self.assertIsInstance(clusters, dict)

    def test_cluster_trajectories_from_observation_euclidean(self):
        clusters, linkage_matrix = cluster_trajectories_from_observation(
            self.observation_trajectories, "euclidean")
        self.assertIsInstance(clusters, dict)

    def test_cluster_trajectories_from_observation_cosine(self):
        clusters, linkage_matrix = cluster_trajectories_from_observation(
            self.observation_trajectories, "cosine")
        self.assertIsInstance(clusters, dict)

    def test_cluster_full_trajectories_euclidean(self):
        clusters, linkage_matrix = cluster_full_trajectories(
            self.full_trajectories, "euclidean")
        self.assertIsInstance(clusters, dict)

    def test_cluster_full_trajectories_cosine(self):
        clusters, linkage_matrix = cluster_full_trajectories(self.full_trajectories, "cosine")
        self.assertIsInstance(clusters, dict)


if __name__ == "__main__":
    unittest.main()
