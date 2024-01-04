import random
import numpy as np
from typing import Tuple, List, Dict


class Neuron:
    """A Neuron class representing a complex single neuron in a neural network.

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

    def __init__(self, layer: int) -> None:
        """Initialize the Neuron with random weights, bias, and specified dimensions and layer.

        Args:
            layer (int): The layer number the neuron is part of.
        """
        self.weights = np.random.rand(3, 2)
        self.inputs = np.zeros((3, 2))
        self.bias = random.random()
        self.edges = list()
        self.layer = layer
        self.metrics = {}  # Empty dictionary for metrics
        self.cosine = False

    def activate(self, inputs: np.ndarray, cosine=False) -> np.ndarray:
        """Compute the neuron's output using a simple linear activation.

        Args:
            inputs (np.ndarray): The inputs to the neuron.
            where I = 2x3 matrix. cols: (input, output), rows: examples

        Returns:
            np.ndarray: The output of the neuron after applying the weights, bias, and activation function.
        """
        self.cosine = cosine
        self.inputs = inputs
        # Simple linear activation: weights * inputs + bias
        if self.cosine:
            return np.cos(self.weights, inputs.T) + self.bias
        return np.dot(self.weights, inputs) + self.bias

    def derivative(self, outputs: np.ndarray,):
        '''
        returns the derivative of the activation function
        '''
        if not self.cosine:
            return 1/1 + np.exp(-1 * self.weights)
        return -(np.sin(self.weights))


if __name__ == "__main__":
    # Example usage:
    neuron = Neuron(layer=1)
    print('activation function')
    print(neuron.activate(np.random.rand(2, 3), True))
    # print(f'weights: {neuron.weights}')
    print('derivative of activation function')
    print(neuron.derivative())
    # print(f'weights: {neuron.weights}')
