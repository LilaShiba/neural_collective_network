import matplotlib.pyplot as plt
import unittest
import numpy as np
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


def main():
    # Generate a noisy sine wave
    sine_wave_data = Vector.generate_noisy_sin()
    vector = Vector(data_points=sine_wave_data)

    # Plot the Probability Density Function (PDF)
    print("Plotting PDF...")
    vector.plot_pdf()

    # Plot basic statistics: mean and standard deviation
    print("Plotting basic statistics...")
    vector.plot_basic_stats()

    # Calculate and print rolling average with a window size of 5
    rolling_avg = vector.rolling_average(window_size=5)
    print("Rolling average (first 10 values):", rolling_avg[:10])

    # Get and print probability vector for x-axis
    prob_vector_x = vector.get_prob_vector(axis=0, rounding=2)
    print("Probability vector for x-axis (first 5):",
          dict(list(prob_vector_x.items())[:5]))

    # If y-axis exists, get and print probability vector for y-axis
    if vector.y is not None:
        prob_vector_y = vector.get_prob_vector(axis=1, rounding=2)
        print("Probability vector for y-axis (first 5):",
              dict(list(prob_vector_y.items())[:5]))

    # Demonstrate set operations using two vectors
    sine_wave_data_2 = Vector.generate_noisy_sin()
    vector2 = Vector(data_points=sine_wave_data_2)
    union, intersection, jaccard_index = Vector.set_operations(vector, vector2)
    print(f"Union (sample): {list(union)[:5]}")
    print(f"Intersection (sample): {list(intersection)[:5]}")
    print(f"Jaccard Index: {jaccard_index}")

    # Calculate entropy between two probability distributions (example)
    prob_dist1 = np.array(list(prob_vector_x.values()))
    prob_dist2 = np.array(list(prob_vector_y.values()))
    entropy = Vector.calculate_aligned_entropy(vector, vector2)
    print(f"Entropy between two probability distributions: {entropy}")


if __name__ == '__main__':
    # Sunittest.main()
    main()
