import unittest
import numpy as np
# Make sure to replace 'your_vector_class_file' with the actual file name where your Vector class is defined.
from vectors import Vector


class TestVector(unittest.TestCase):

    def setUp(self):
        # Create a simple vector for testing
        self.data = Vector.generate_noisy_sin(points=100)
        self.vector = Vector(label=1, data_points=self.data)

    def test_initialization(self):
        """Test initialization of Vector class."""
        self.assertEqual(self.vector.label, 1)
        self.assertEqual(len(self.vector.x), 100)
        self.assertIsNotNone(self.vector.v)

    def test_prob_vector(self):
        """Test probability vector calculation."""
        prob_vector = self.vector.get_prob_vector()
        # Check if the sum of probabilities is close to 1
        self.assertAlmostEqual(sum(prob_vector.values()), 1.0, places=2)

    def test_rolling_average(self):
        """Test rolling average calculation."""
        rolling_avg = self.vector.rolling_average(window_size=5)
        # Check if the rolling average reduces the size of the array correctly
        self.assertEqual(len(rolling_avg), len(self.vector.x) - 4)

    def test_set_operations(self):
        """Test set operations: union, intersection, and Jaccard index."""
        # Create another vector
        data2 = Vector.generate_noisy_sin(start=1, points=100)
        vector2 = Vector(label=2, data_points=data2)

        union, intersection, jaccard_index = Vector.set_operations(
            self.vector, vector2)
        # Check if union and intersection are sets
        self.assertIsInstance(union, set)
        self.assertIsInstance(intersection, set)
        # Jaccard index should be between 0 and 1
        self.assertTrue(0 <= jaccard_index <= 1)

    def test_entropy(self):
        """Test entropy calculation between two probability distributions."""
        # Create another vector
        data2 = Vector.generate_noisy_sin(start=1, points=100)
        vector2 = Vector(label=2, data_points=data2)

        prob_dist1 = np.array(list(self.vector.get_prob_vector().values()))
        prob_dist2 = np.array(list(vector2.get_prob_vector().values()))

        # Ensure the probability distributions have the same length
        min_length = min(len(prob_dist1), len(prob_dist2))
        prob_dist1 = prob_dist1[:min_length]
        prob_dist2 = prob_dist2[:min_length]

        ent = Vector.calculate_entropy(prob_dist1, prob_dist2)
        # Entropy should be a non-negative value
        self.assertGreaterEqual(ent, 0)


if __name__ == '__main__':
    unittest.main()
