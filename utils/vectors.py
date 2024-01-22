import numpy as np
import matplotlib.pyplot as plt
from typing import *
from collections import defaultdict


class Vector:
    ''' Vector Maths Class'''

    def __init__(self, label: int = 0, data_points: np.array = None):
        self.label = label
        if data_points is not None:
            self.v = data_points
            self.n = len(data_points)
            self.x = data_points[:, 0]
            if data_points.ndim == 2:
                self.y = data_points[:-1]

    def count(self, array: np.array, value: float):
        '''returns count of value in an array'''
        return len(array[array == value])

    def get_prob_vectors(self, rounding=None):
        '''run get_prob_vector for all axis'''
        return self.get_prob_vector(0, rounding), self.get_prob_vector(1)

    def get_prob_vector(self, axis=0, rounding=None):
        '''return prob vector
        axis 0 = x
        axis 1 = y
        '''

        if axis == 0:
            vector = self.x
        else:
            vector = self.y
        single_values = set(vector)
        if rounding is not None:
            vector = np.array([round(num, rounding) for num in single_values])

        single_values = set(vector)
        # print(len(single_values), len(vector))

        self.prob_dict = {data_point: self.count(
            vector, data_point)/self.n for data_point in single_values}

        return self.prob_dict

    def savitzky_golay_filter(self, window=3):
        '''returns vector after smoothing with savitzky-golay filter'''
        pass

    @staticmethod
    def generate_nosiey_sin(start=0, points=100):
        ''' create a sine wave for testing'''
        # Step 1: Generate Sample Data
        x = np.linspace(start, 2 * np.pi, points)  # 100 points from 0 to 2π
        y = np.sin(x) + np.random.normal(start, 0.2, points)
        return np.array(list(zip(x, y)))


if __name__ == "__main__":
    data = Vector.generate_nosiey_sin()
    v1 = Vector(label=0, data_points=data)
    d = v1.get_prob_vector(rounding=2)
