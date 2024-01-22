import numpy as np
from typing import *
import matplotlib.pyplot as plt
from neurons import Neuron
from vectors import Vector


class Layer:
    '''Linear applications of vectors consisting of nodes'''

    def __init__(self, data_points: np.array, layer_number: int = 0, name: str = "Input_Layer", number_of_neurons: int = None, weights: np.array = None):

        self.x: np.array = data_points[0]
        self.n: int = len(self.x)
        self.y: np.array = data_points
        self.m: int = len(self.y)
        self.db: np.array(np.array) = data_points
        self.name: str = name
        self.neurons: np.array = None
        self.layer_number: int = layer_number
        self.number_of_neurons: int = number_of_neurons
        self.inputs: np.array = None  # vectorized neuron input
        self.weights: np.array = weights  # vectorized neuron weights

    def create_neurons(self, number_of_neurons: int):
        '''
        creates self.neurons: np.array of neurons based on datapoints
        creates self.weights: np.array of 2x3 weights based on neurons


        '''
        self.neurons = np.array([Neuron([self.x[idx], self.y[idx]], layer=1, label=self.name)
                                 for idx in range(number_of_neurons)])
        self.weights = np.array([neuron.weights for neuron in self.neurons])
        self.inputs = np.array([[neuron.x, neuron.state, 1]
                               for neuron in self.neurons])
        return self.inputs, self.weights

    def feed_forward(self):
        '''
        vectorized feed forward operation of neruons
        '''
        self.signal = np.tanh(
            (self.inputs * self.weights))  # + self.bias
        self.state, self.output = self.signal[0], self.signal[1]
        self.last_input = self.signal
        return [self.state, self.output]


if __name__ == "__main__":
    sine_wave = np.array(Vector.generate_nosiey_sin())

    layer_input = Layer(sine_wave)
    inputs, weights = layer_input.create_neurons(len(sine_wave))
    state, output = layer_input.feed_forward()
    print(state)
