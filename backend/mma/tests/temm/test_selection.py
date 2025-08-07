import unittest
import numpy as np

from mma_wrapper.temm.selection import select_near_centroid_trajectories

class TestSelection(unittest.TestCase):

    def test_select_near_centroid_vector(self):
        # Cluster 0 has three trajectories
        clusters = {
            0: [
                [np.array([1.0, 1.0]), np.array([2.0, 2.0])],
                [np.array([1.1, 1.1]), np.array([2.1, 2.1])],
                [np.array([5.0, 5.0]), np.array([6.0, 6.0])]
            ]
        }

        # Centroid close to the first two
        centroids = {
            0: [np.array([1.05, 1.05]), np.array([2.05, 2.05])]
        }

        selected = select_near_centroid_trajectories({}, 0,
            clusters[0], centroids, inclusion_radius=0.5)

        self.assertIn(0, selected)
        self.assertEqual(len(selected[0]), 2)

    def test_select_all_if_radius_high(self):
        clusters = {
            0: [
                [np.array([1.0]), np.array([2.0])],
                [np.array([5.0]), np.array([6.0])]
            ]
        }
        centroids = {
            0: [np.array([3.0]), np.array([4.0])]
        }

        selected = select_near_centroid_trajectories({}, 0,
            clusters[0], centroids, inclusion_radius=10.0)

        self.assertEqual(len(selected[0]), 2)

    def test_select_none_if_radius_low(self):
        clusters = {
            0: [
                [np.array([1.0]), np.array([2.0])],
                [np.array([5.0]), np.array([6.0])]
            ]
        }
        centroids = {
            0: [np.array([3.0]), np.array([4.0])]
        }

        selected = select_near_centroid_trajectories({}, 0,
            clusters[0], centroids, inclusion_radius=0.01)

        self.assertEqual(len(selected[0]), 0)

if __name__ == '__main__':
    unittest.main()
