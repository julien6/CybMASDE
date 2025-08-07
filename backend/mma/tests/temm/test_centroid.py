import unittest
import numpy as np

from mma_wrapper.temm.centroid import compute_centroids_per_cluster


class TestCentroidComputation(unittest.TestCase):

    def test_mean_centroid_on_vector_trajectories(self):
        clusters = {
            0: [
                [np.array([1, 2]), np.array([3, 4])],
                [np.array([2, 3]), np.array([4, 5])]
            ],
            1: [
                [np.array([0, 0]), np.array([1, 1])]
            ]
        }

        centroids = compute_centroids_per_cluster(clusters, mode="mean")

        self.assertIn(0, centroids)
        self.assertIn(1, centroids)
        self.assertEqual(len(centroids[0]), 2)

        np.testing.assert_almost_equal(centroids[0][0], np.array([1.5, 2.5]))
        np.testing.assert_almost_equal(centroids[0][1], np.array([3.5, 4.5]))

    def test_fuzzy_lcs_centroid(self):
        clusters = {
            0: [
                [1, 2, 3, 4],
                [1, 2, 4, 5],
                [1, 3, 4, 6],
            ]
        }

        centroids = compute_centroids_per_cluster(clusters, mode="fuzzy_lcs")
        self.assertIsInstance(centroids[0], list)
        self.assertTrue(all(isinstance(a, int) for a in centroids[0]))
        self.assertTrue(len(centroids[0]) > 0)

    def test_invalid_mode_raises(self):
        clusters = {0: [[1, 2, 3]]}
        with self.assertRaises(ValueError):
            compute_centroids_per_cluster(clusters, mode="invalid")


if __name__ == "__main__":
    unittest.main()
