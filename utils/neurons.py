import numpy as np
from typing import *
import matplotlib.pyplot as plt
from utils.vectors import Vector


class Neuron():
    def __init__(self, inputs: np.array, layer: int, label: str = 'test'):
        self.layer: int = layer
        self.label: str = label
        self.x: float = inputs[0]
        self.y: float = inputs[1]
        self.weights: np.array(np.array) = np.random.rand(
            2, 3)  # 2x3 matrix of random floats
        self.bias: float = np.random.random()
        self.learning_rate: float = np.random.random()
        self.state: float = np.random.random()
        self.output: float = np.random.random()
        self.loss_gradient: np.array = None
        self.last_input: np.array = None
        self.signal: np.array = None

    def compute_gradient(self, output):
        '''distance between predictions and ground truth'''
        error = output - self.y
        derivative = 1 - np.tanh(output) ** 2
        gradient = error * derivative * self.x
        return gradient, error

    def feed_forward(self, input_vector=None) -> np.ndarray:
        """Compute the neuron's output using a simple linear activation.

        Args:
            inputs (np.ndarray): The inputs to the neuron.
            where I = 3x2 matrix. cols: (input, output), rows: examples

        Returns:
            np.ndarray: The output of the neuron after applying the weights, bias, and activation function.
        """
        if input_vector is None:
            input_vector = [self.x, self.state, 1]
        # delta = (np.dot(self.inputs, self.weights.T) + self.bias).T
        self.signal = np.tanh(np.dot(input_vector, self.weights.T)) + self.bias
        self.state, self.output = self.signal[0], self.signal[1]
        self.last_input = self.signal
        return [self.state, self.output]

    def derivative(self):
        '''
        returns the derivative of the activation function
        '''
        return (1.0 - np.tanh(self.signal) ** 2)
        # return self.sigmoid(self.delta) * (1-self.sigmoid(self.delta))

    def iterate(self):
        """
        Compute the gradient of the neuron's weights with respect to the loss.

        :param neuron_loss_grad: The gradient of the loss with respect to the neuron's output.
        :return: The neuron signal after 1 iteration.
        """
        predictions = self.feed_forward()
        gradient, error = self.compute_gradient(predictions[1])
        self.weights -= self.learning_rate * gradient
        self.bias -= self.learning_rate * error
        return self

    @staticmethod
    def sigmoid(x):
        """The Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of the Sigmoid function."""
        return Neuron.sigmoid(x) * (1 - Neuron.sigmoid(x))


if __name__ == "__main__":
    # Example usage:
    print('begin test')
    sine_wave = np.array(Vector.generate_nosiey_sin())
    one_point = np.array([sine_wave[0][0], sine_wave[1][0]])
    neuron = Neuron(inputs=one_point, layer=1)
    for x in range(100):
        neuron.iterate()
        if x % 10 == 0:
            print(
                f" iteratrion:{x} state:{neuron.state} output:{neuron.output}")
            print('')

    # neuron.iterate()
    # print(neuron.weights)

    # n2 = Neuron(inputs=[1, 2], layer=2, weights=neuron.weights)
    # print(n2.feed_forward())
    # # print(n2.weights)
    # print('')
    # print(n2.state)
    print(f' prediction: {neuron.output} res: {neuron.y}')
