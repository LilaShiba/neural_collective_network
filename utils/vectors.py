import numpy as np
import matplotlib.pyplot as plt
from typing import *
import collections


class Vector:
    '''Vector Maths Class for statistical analysis and visualization.'''

    def __init__(self, label: int = 0, data_points: np.array = None):
        self.label = label
        if data_points is not None:
            self.v = data_points
            self.n = len(data_points)

            if data_points.ndim == 2 and data_points.shape[1] > 1:
                self.x = data_points[:, 0]
                self.y = data_points[:, 1]
            else:
                self.x = data_points
                self.y = None

    def count(self, array: np.array, value: float) -> int:
        '''Returns count of value in an array.'''
        return len(array[array == value])

    def linear_scale(self):
        histo_gram = collections.Counter(self.x)
        val, cnt = zip(*histo_gram.items())

        n = len(cnt)
        prob_vector = [x / n for x in cnt]
        plt.plot(val, prob_vector, 'x')
        plt.show()

    def log_binning(self) -> Tuple[float, float]:
        """Plot the degree distribution with log binning."""
        histo_gram = collections.Counter(self.x)
        val, cnt = zip(*histo_gram.items())

        n = len(cnt)
        prob_vector = [x / n for x in cnt]
        in_max, in_min = max(prob_vector), min(prob_vector)
        log_bins = np.logspace(np.log10(in_min), np.log10(in_max))
        deg_hist, log_bin_edges = np.histogram(
            prob_vector, bins=log_bins, density=True, range=(in_min, in_max))
        plt.title(f"Log Binning & Scaling")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(deg_hist, log_bin_edges[:-1], 'o')
        plt.show()
        return in_min, in_max

    def get_prob_vector(self, axis: int = 0, rounding: int = None) -> Dict[float, float]:
        '''Return probability vector for a given axis.'''
        if axis == 0:
            vector = self.x
        elif self.y is not None:
            vector = self.y
        else:
            raise ValueError("Invalid axis for probability vector.")

        if rounding is not None:
            vector = np.round(vector, rounding)

        unique_values = np.unique(vector)
        prob_dict = {value: self.count(
            vector, value) / self.n for value in unique_values}
        return prob_dict

    def plot_pdf(self, bins: int = 'auto'):
        '''Plots the Probability Density Function (PDF) of the vector.'''
        if self.y is not None:
            data = self.y
        else:
            data = self.x

        density, bins, _ = plt.hist(
            data, bins=bins, density=True, alpha=0.5, label='PDF')
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title('Probability Density Function')
        plt.legend()
        plt.show()

    def plot_basic_stats(self):
        '''Plots basic statistics: mean and standard deviation.'''
        if self.y is not None:
            data = self.y
        else:
            data = self.x

        mean = np.mean(data)
        std = np.std(data)

        plt.hist(data, bins='auto', alpha=0.5, label='Data')
        plt.axvline(mean, color='r', linestyle='dashed',
                    linewidth=1, label=f'Mean: {mean:.2f}')
        plt.axvline(mean + std, color='g', linestyle='dashed',
                    linewidth=1, label=f'Std: {std:.2f}')
        plt.axvline(mean - std, color='g', linestyle='dashed', linewidth=1)
        plt.legend()
        plt.show()

    def rolling_average(self, window_size: int = 3) -> np.array:
        '''Calculates and returns the rolling average of the vector using numpy.'''
        if self.y is not None:
            data = self.y
        else:
            data = self.x

        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

    @staticmethod
    def calculate_aligned_entropy(vector1, vector2) -> float:
        """
        Calculates the entropy between two aligned probability distributions from Vector instances.
        """

        # Create probability distributions with zeros for missing values in each vector
        prob_dist1 = np.array(list(vector1.get_prob_vector().values()))
        prob_dist2 = np.array(list(vector2.get_prob_vector().values()))

        # Ensure the probability distributions are normalized
        # prob_dist1 = prob_dist1 / np.sum(prob_dist1)
        # prob_dist2 = prob_dist2 / np.sum(prob_dist2)
        # Calculate joint probabilities
        joint_probs = prob_dist1 * prob_dist2

        # Filter out zero probabilities to avoid NaNs in the logarithm
        joint_probs = joint_probs[joint_probs != 0]

        # Calculate entropy
        entropy = -np.sum(joint_probs * np.log2(joint_probs))

        # Calculate the entropy
        return entropy

    @staticmethod
    def set_operations(v1: 'Vector', v2: 'Vector') -> Tuple[Set[float], Set[float], float]:
        '''Performs set operations: union, intersection, and calculates Jaccard index.'''
        set1 = set(v1.x)
        set2 = set(v2.y)

        union = set1.union(set2)
        intersection = set1.intersection(set2)
        jaccard_index = len(intersection) / len(union)

        return union, intersection, jaccard_index

    @staticmethod
    def generate_noisy_sin(start: float = 0, points: int = 100) -> Tuple[np.array, np.array]:
        '''Creates a noisy sine wave for testing.'''
        x = np.linspace(start, 2 * np.pi, points)
        y = np.sin(x) + np.random.normal(0, 0.2, points)
        return np.column_stack((x, y))


if __name__ == "__main__":
    # Set parameters
    mean = 50
    std_dev = 10
    sample_size = 2000

    # Generate the sample list
    s1 = np.round(np.random.normal(mean, std_dev, sample_size), 2)
    s2 = np.round(np.random.normal(mean, std_dev, sample_size), 2)

    v1 = Vector('1', s1)
    v2 = Vector('2', s2)
    v1.linear_scale()
    v1.log_binning()
