import numpy as np
from typing import *
import matplotlib.pyplot as plt
from neurons import Neuron
from vectors import Vector


class Layer:
    '''Linear applications of vectors consisting of complex neurons'''

    def __init__(self, data_points: np.array, layer_number: int = 0, name: str = "Input_Layer", number_of_neurons: int = None, weights: np.array = None):

        self.x: np.array = data_points[0]
        self.n: int = len(self.x)
        self.y: np.array = data_points[1]
        self.m: int = len(self.y)
        self.db: np.array(np.array) = data_points
        self.name: str = name
        self.state: np.array = None
        self.output: np.array(np.array) = None
        self.signal: np.array(np.array) = None

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
        return [self.state, self.output]

    def train(self, epochs: int = 10):
        '''
        vectorized back_propagation using tanh
        '''
        vectorized_iterate = np.vectorize(lambda neuron: neuron.iterate())

        for x in range(epochs):
            self.neurons = vectorized_iterate(self.neurons)
        return self.neurons


if __name__ == "__main__":
    sine_wave = np.array(Vector.generate_nosiey_sin())

    layer_input = Layer(sine_wave)
    inputs, weights = layer_input.create_neurons(len(sine_wave[0]))
    # state, output = layer_input.feed_forward()
    # print(layer_input.neurons[0].weights)
    vect_neurons = layer_input.train(epochs=12)
    print(
        f' prediction: {vect_neurons[0].output} ground truth: {vect_neurons[0].y}')
