import random
import numpy as np
from typing import *
import matplotlib.pyplot as plt


class Neuron:
    """A class representing a complex single neuron in a neural network.
s (Dict[str, float]): A dictionary for storing various performance metrics.
    """

    def __init__(self, inputs: np.array(np.array), layer: int, weights: np.array = []) -> None:
        """Initialize the Neuron with random weights, bias, and specified dimensions and layer.

        Args:
            layer (int): The layer number the neuron is part of.
        """
        self.weights = weights if len(weights) > 0 else np.random.rand(3, 2)
        self.learning_rate: float = random.uniform(0.1, 1)
        self.inputs = np.array(inputs)
        self.inputs_x = self.inputs[0]
        self.inputs_y = self.inputs[-1]
        self.bias = np.random.random(1)[0]  # random.uniform(0.1, 0.5)
        self.edges = list()
        self.layer: int = layer
        self.delta: np.array = None  # activation function output
        self.loss_gradient: np.array = None
        self.last_input: np.array = self.inputs
        self.state = np.random.lognormal(0, 1, 1)[0]
        self.signal: float = np.random.lognormal(0, 1, 1)[0]

    def activate(self, inputs=False) -> np.ndarray:
        """Compute the neuron's output using a simple linear activation.

        Args:
            inputs (np.ndarray): The inputs to the neuron.
            where I = 3x2 matrix. cols: (input, output), rows: examples

        Returns:
            np.ndarray: The output of the neuron after applying the weights, bias, and activation function.
        """
        if inputs:
            self.inputs = inputs
        # Simple linear/non-linear activation: weights * inputs + bias
        self.last_input = self.inputs
        # delta = (np.dot(self.inputs, self.weights.T) + self.bias).T

        self.delta = np.tanh(
            np.dot([self.inputs_x, self.state, 1], self.weights) + self.bias)
        self.state, self.signal = self.delta[0], self.delta[1]
        return self.signal

    def derivative(self):
        '''
        returns the derivative of the activation function
        '''
        return (1.0 - np.tanh(self.delta) ** 2)
        # return self.sigmoid(self.delta) * (1-self.sigmoid(self.delta))

    def iterate(self):
        """
        Compute the gradient of the neuron's weights with respect to the loss.

        :param neuron_loss_grad: The gradient of the loss with respect to the neuron's output.
        :return: The gradient with respect to the neuron's weights.
        """
        self.activate()
        res = self.derivative()
        neuron_loss_grad = (self.activate() - self.inputs_y)
        d_output_d_weights = res * self.last_input.T
        gradient = neuron_loss_grad * d_output_d_weights
        self.loss_gradient = gradient
        self.weights -= (self.learning_rate * gradient.T)
        return gradient

    def update_weights(self, neuron_weight_grad: np.array):
        '''
        Update weights in backpropagation abstracted in Layer class
        '''
        self.weights -= (self.learning_rate * self.loss_gradient.T)
        # TODO update bias during backpropagation
        # self.bias -= self.learning_rate * bias_gradient

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
    neuron = Neuron(inputs=np.array([1, 2]), layer=1)
    print(neuron.iterate())
