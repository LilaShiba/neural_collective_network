import numpy as np
from typing import *
import matplotlib.pyplot as plt


class Neuron:
    """A class representing a complex single neuron in a neural network.
s (Dict[str, float]): A dictionary for storing various performance metrics.
    """

    def __init__(self, inputs: np.array(np.array), layer: int, weights: np.array = None) -> None:
        """Initialize the Neuron with random weights, bias, and specified dimensions and layer.

        Args:
            layer (int): The layer number the neuron is part of.
        """
        self.weights: np.array = np.random.rand(3, 2)
        self.learning_rate: float = np.random.random(1)[0]
        self.inputs_x = inputs[0]
        self.inputs_y = inputs[-1]
        self.bias = np.random.random(1)[0]  # random.uniform(0.1, 0.5)
        self.edges = list()
        self.layer: int = layer
        self.delta: np.array = None  # activation function output
        self.loss_gradient: np.array = None
        self.last_input: np.array = self.inputs_x
        self.state: float = np.random.random(
            1)[0]  # np.random.lognormal(0, 1, 1)[0]
        self.signal: np.array = np.random.rand(3)

    def compute_gradient(self, neuron_loss_grad: np.array):
        """
        Compute the gradient of the neuron's weights with respect to the loss.

        :param neuron_loss_grad: The gradient of the loss with respect to the neuron's output.
        :return: The gradient with respect to the neuron's weights.
        """
        derivative = self.derivative()
        d_output_d_weights = derivative * self.last_input.T
        gradient = neuron_loss_grad * d_output_d_weights
        self.loss_gradient = gradient
        return gradient

    def feed_forward(self, inputs=False) -> np.ndarray:
        """Compute the neuron's output using a simple linear activation.

        Args:
            inputs (np.ndarray): The inputs to the neuron.
            where I = 3x2 matrix. cols: (input, output), rows: examples

        Returns:
            np.ndarray: The output of the neuron after applying the weights, bias, and activation function.
        """

        # delta = (np.dot(self.inputs, self.weights.T) + self.bias).T

        self.delta = np.tanh(self.signal
                             * self.weights.T) + self.bias
        self.state, self.signal = self.delta[0], self.delta[1]
        self.last_input = self.signal
        return self.delta

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
        :return: The neuron signal after 1 iteration.
        """
        delta_signal = self.feed_forward()
        res = self.derivative()
        neuron_loss_grad = (delta_signal - self.inputs_y)
        d_output_d_weights = res * self.inputs_y
        gradient = neuron_loss_grad * d_output_d_weights

        self.weights -= (self.learning_rate * gradient).T
        self.loss_gradient = gradient
        return self.signal

    def update_weights(self, neuron_weight_grad):
        '''
        Update weights in backpropagation abstracted in Layer class
        '''
        self.weights -= (self.learning_rate * neuron_weight_grad).T
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
    neuron = Neuron(inputs=np.array([1, 2 * np.pi]), layer=1)
    print(neuron.iterate())
    print(neuron.weights)

    n2 = Neuron(inputs=[1, 2], layer=2, weights=neuron.weights)
    print(n2.feed_forward())
    # print(n2.weights)
    print('')
    print(n2.state)
