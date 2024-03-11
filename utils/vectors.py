import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Set


class Vector:
    '''Vector Maths Class for statistical analysis and visualization.'''

    def __init__(self, label: int = 0, data_points: np.array = None):
        self.label = label
        if data_points is not None:
            self.v = data_points
            self.n = len(data_points)
            self.x = data_points[:, 0]
            if data_points.ndim == 2 and data_points.shape[1] > 1:
                self.y = data_points[:, 1]
            else:
                self.y = None

    def count(self, array: np.array, value: float) -> int:
        '''Returns count of value in an array.'''
        return len(array[array == value])

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
    def calculate_entropy(prob_dist1: np.array, prob_dist2: np.array) -> float:
        '''Calculates the entropy between two probability distributions.'''
        # Ensure the probability distributions are normalized
        prob_dist1 = prob_dist1 / np.sum(prob_dist1)
        prob_dist2 = prob_dist2 / np.sum(prob_dist2)

        # Calculate the entropy
        return -np.sum(prob_dist1 * np.log(prob_dist2 / prob_dist1))

    @staticmethod
    def set_operations(v1: 'Vector', v2: 'Vector') -> Tuple[Set[float], Set[float], float]:
        '''Performs set operations: union, intersection, and calculates Jaccard index.'''
        set1 = set(v1.x)
        set2 = set(v2.x)

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
