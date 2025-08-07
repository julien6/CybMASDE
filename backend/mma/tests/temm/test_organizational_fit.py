# test_organizational_fit.py

import unittest
import numpy as np
from mma_wrapper.temm.organizational_fit import compute_sof, compute_fof, compute_overall_fit


class TestOrganizationalFit(unittest.TestCase):

    def setUp(self):
        # Mock cluster with trajectories: each trajectory is a list of one-hot vectors
        self.mock_clusters = {
            0: [[np.array([[1, 0], 1]), np.array([[1, 0], 1])]],
            1: [[np.array([[0, 1], 0]), np.array([[0, 1], 0])]]
        }

        # Centroids are mean trajectories (same shape as individual trajectories)
        self.mock_centroids = {
            0: np.array([1, 0, 0, 1, 1, 0, 0, 1]),
            1: np.array([0, 1, 1, 0, 0, 1, 1, 0])
        }

        self.mock_observation_clusters = {
            0: [[np.array([1, 0, 1, 0])], [np.array([2, 0, 2, 0])]],
            1: [[np.array([0, 1, 0, 1])]]
        }

        self.mock_observation_centroids = {
            0: np.array([1.5, 0, 1.5, 0]),
            1: np.array([0, 1, 0, 1])
        }

    def test_compute_sof(self):
        sof = compute_sof(self.mock_clusters, self.mock_centroids)
        self.assertIsInstance(sof, float)
        self.assertGreaterEqual(sof, 0.0)
        self.assertLessEqual(sof, 1.0)

    def test_compute_fof(self):
        fof = compute_fof(self.mock_observation_clusters,
                          self.mock_observation_centroids)
        self.assertIsInstance(fof, float)
        self.assertGreaterEqual(fof, 0.0)
        self.assertLessEqual(fof, 1.0)

    def test_compute_overall_fit(self):
        sof = 0.9
        fof = 0.8
        of = compute_overall_fit(sof, fof, aggregation_method="mean")
        self.assertAlmostEqual(of, 0.85)

        of_sum = compute_overall_fit(sof, fof, aggregation_method="sum")
        self.assertEqual(round(of_sum, 2), 1.7)


if __name__ == "__main__":
    unittest.main()
