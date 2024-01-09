import random
import numpy as np
from typing import Tuple, List, Dict


class Neuron:
    """A class representing a complex single neuron in a neural network.

    W = 2x3 
    cols: input, output
    rows: examples

    Attributes:
        weights (np.ndarray): A 2x3 matrix with random weights.
        bias (float): A random float representing the bias.
        edges (List[int]): A sorted list of edges, initialized as empty.
        layer (int): An integer representing the layer number the neuron belongs to. Layer is a 
            class where input size matches neuron amount
        metrics (Dict[str, float]): A dictionary for storing various performance metrics.
    """

    def __init__(self, inputs: np.array(np.array), layer: int, tanh=True, weights: np.array = []) -> None:
        """Initialize the Neuron with random weights, bias, and specified dimensions and layer.

        Args:
            layer (int): The layer number the neuron is part of.
        """
        self.weights = weights if len(weights) > 0 else np.random.rand(3, 2)
        self.learning_rate: float = random.uniform(0.1, 1)
        self.inputs = np.array(inputs)
        self.inputs_x = self.inputs[:, 0]
        self.inputs_y = self.inputs[:, -1]
        self.bias = random.uniform(0.1, 0.5)
        self.edges = list()
        self.layer: int = layer
        self.metrics = dict()
        self.tanh: bool = tanh
        self.delta: np.array = None  # activation function output
        self.loss_gradient: np.array = None
        self.last_input: np.array = None

    def activate(self, tanh=False) -> np.ndarray:
        """Compute the neuron's output using a simple linear activation.

        Args:
            inputs (np.ndarray): The inputs to the neuron.
            where I = 3x2 matrix. cols: (input, output), rows: examples

        Returns:
            np.ndarray: The output of the neuron after applying the weights, bias, and activation function.
        """
        self.tanh = tanh
        # Simple linear/non-linear activation: weights * inputs + bias
        self.last_input = self.inputs
        delta = (np.dot(self.inputs, self.weights.T) + self.bias).T
        if self.tanh:
            delta = np.tanh(delta)
        else:
            delta = self.sigmoid(delta)
        self.delta = delta
        return np.mean(self.delta)

    def derivative(self):
        '''
        returns the derivative of the activation function
        '''
        if self.tanh:
            return (1.0 - np.tanh(self.delta) ** 2)
        return self.sigmoid(self.delta) * (1-self.sigmoid(self.delta))

    def compute_gradient(self, neuron_loss_grad: np.array):
        """
        Compute the gradient of the neuron's weights with respect to the loss.

        :param neuron_loss_grad: The gradient of the loss with respect to the neuron's output.
        :return: The gradient with respect to the neuron's weights.
        """
        res = self.derivative()
        res = res[:-1, :]
        d_output_d_weights = res * self.last_input.T
        gradient = neuron_loss_grad * d_output_d_weights
        self.loss_gradient = gradient
        return gradient

    def update_weights(self, neuron_weight_grad: np.array):
        '''
        Update weights in backpropagation abstracted in Layer class
        '''
        self.weights -= (self.learning_rate * neuron_weight_grad.T)
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
    neuron = Neuron(inputs=np.random.rand(3, 2), layer=1)
    print('activation function')
    print(neuron.activate(False))
    # print(f'weights: {neuron.weights}')
    print('derivative of activation function')
    print(neuron.derivative())
    # print(f'weights: {neuron.weights}')
