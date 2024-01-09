import random
import numpy as np
from typing import *
from utils.neuron import Neuron
import matplotlib.pyplot as plt


class Layer:
    '''
    Class representing the temporal steps & connections between complex neurons

    '''

    def __init__(self, input=np.array(np.array), weights: np.array(np.array) = []) -> None:
        self.input = input
        self.layer: int = 0
        self.neurons: dict = dict()
        self.loss_grad: np.array = None
        self.learning_rate = 0.1
        if len(weights) > 0:
            self.weights = weights
        else:
            self.weights = []

    def create_neurons(self, group_size: int = 3, random_activation: bool = False, weight=[]):
        '''
        Initialzes neurons in circular fashion
        2x3s
        '''
        n = len(self.input)
        p_of_x = True  # TODO update to allow for neurons with diverse activations
        if random_activation:
            p_of_x = np.random.choice([0, 1])
        for i in range(n):
            # circular array
            delta_group = [self.input[(i + j) % n] for j in range(group_size)]
            if len(self.weights) > 0:
                weight = self.weights[i-1]

            self.neurons[i] = Neuron(
                inputs=delta_group, layer=self.layer, tanh=p_of_x, weights=weight)

    def feed_forward(self):
        '''
        Activate neurons based on their inputs
        '''
        outputs = [neuron.activate(tanh=neuron.tanh)
                   for neuron in self.neurons.values()]
        return outputs

    def back_propagation(self):
        '''
        Backpropagate the error and update the weights of each neuron.
        '''
        # Ensure that the necessary attributes are set
        if self.loss_grad is None:
            raise ValueError(
                "Loss gradient must be set before backpropagation.")
        if self.learning_rate is None:
            raise ValueError(
                "Learning rate must be set before backpropagation.")

        input_grad = np.zeros_like(self.input)

        for idx, neuron in enumerate(self.neurons.values()):
            neuron_loss_grad = self.loss_grad[idx]
            neuron_weight_grad = neuron.compute_gradient(neuron_loss_grad)
            # Update neuron weights
            neuron.update_weights(neuron_weight_grad)

    def get_loss_vector(self, predictions: np.array, targets: np.array):
        '''
        # For Mean Squared Error (MSE):
        # Loss = (1/N) * sum((y_i - y_hat_i)^2) for i = 1 to N

        # For Cross-Entropy:
        # Loss = -sum(y_i * log(y_hat_i)) for i = 1 to N

        '''
        return np.mean((predictions - targets)**2)

    def train(self, epochs: int = 101):
        self.train_errors = []
        targets = np.array([sample[1] for sample in self.input])
        for epoch in range(epochs):
            # Start Feed-Forward
            predictions = self.feed_forward()

            # Calculate loss
            mse_loss = self.get_loss_vector(predictions, targets)
            # Calculate the derivative of the loss with respect to the outputs
            loss_grad = 2 * (predictions - targets) / len(targets)
            self.set_loss_grad(loss_grad)
            # Perform the backpropagation step
            self.back_propagation()
            # (Optional) Print the loss to monitor progress
            if epoch % 10 == 0:  # Print every 10 epochs, adjust as needed
                print(f"Epoch {epoch}, Loss: {mse_loss}")
                self.train_errors.append(mse_loss)

    def set_loss_grad(self, loss_grad: np.array):
        '''
        set the loss gradient for the layer to update backprogation
        '''
        self.loss_grad = loss_grad

    def see_loss_grad(self):
        '''
        Graph of loss gradient
        '''
        if len(self.loss_grad) > 0:
            plt.plot(self.train_errors)
            plt.title('Loss Gradient')
            plt.show()
            plt.close()

    def set_learning_rate(self, learning_rate: float = 0.1):
        self.learning_rate = 0.1
