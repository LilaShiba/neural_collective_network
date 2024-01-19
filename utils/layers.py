import random
import numpy as np
from typing import *
from utils.neurons import Neuron
import matplotlib.pyplot as plt


class Layer:
    '''
    Class representing the temporal steps & connections between complex neurons

    '''

    def __init__(self, layer_size: int, input=np.array(np.array), weights: np.array(np.array) = [], layer_num: int = 0) -> None:
        self.input = input
        self.layer: int = layer_num
        self.layer_size: int = layer_size
        self.neurons: dict = dict()
        self.loss_grad: np.array = None
        self.learning_rate = np.random.random(1)[0]
        self.weights = []
        if len(weights) > 0:
            self.weights = weights

    def create_neurons_list(self, group_size: int = 1):
        '''
        Initialzes neurons in circular fashion
        2x3s
        Returns: List of neuron weights
        '''
        db = self.input
        n = len(db)
        # m = len(db[0])
        # m = len(self.weights)
        for i in range(n):

            delta_group = db[i]
            # delta_weights = []
            # if len(self.weights) > 0:
            #     delta_weights = [self.weights[i: (i + j) % m]
            #                      for j in range(group_size)]
            # if len(delta_weights) > 0:
            #     print(f'delta weights {delta_weights}')
            self.neurons[i] = Neuron(
                inputs=delta_group, layer=self.layer, weights=[])
        return [n.weights for n in self.neurons.values()]

    def create_neurons_matrix(self, group_size: int = 1):
        '''
        Initialzes neurons in circular fashion
        2x3s
        Returns: List of neuron weights
        '''
        db = self.input[:, 0]
        n = len(db)
        m = len(db[0])
        # m = len(self.weights)
        for i in range(n):
            for j in range(group_size):
                # circular array
                delta_group = db[i, j]
                # delta_weights = []
                # if len(self.weights) > 0:
                #     delta_weights = [self.weights[i: (i + j) % m]
                #                      for j in range(group_size)]
                # if len(delta_weights) > 0:
                #     print(f'delta weights {delta_weights}')
                self.neurons[(i, j)] = Neuron(
                    inputs=[delta_group, self.input[-1]], layer=self.layer, weights=[])
        return [n.weights for n in self.neurons.values()]

    def iterate(self):
        '''one training cycle for a layer'''
        targets = self.input[-1]
        predictions = self.feed_forward()

        print(predictions)

        mse_loss = self.get_loss_vector(predictions, targets)
        # Calculate the derivative of the loss with respect to the outputs
        # loss_grad = 2 * (predictions - targets) / len(targets)
        self.loss_grad = mse_loss
        # Perform the backpropagation step
        self.back_propagation()

    def feed_forward(self, data=False) -> np.array:
        '''
        Activate neurons based on their inputs
        '''
        outputs = [neuron.feed_forward(input_vector=data)
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
        for idx, neuron in enumerate(self.neurons.values()):
            neuron_loss_grad = self.loss_grad
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
        targets = self.input[1]
        for epoch in range(epochs):
            # Start Feed-Forward
            predictions = self.feed_forward()

            # Calculate loss
            mse_loss = self.get_loss_vector(predictions, targets)
            # Calculate the derivative of the loss with respect to the outputs
            loss_grad = 2 * (predictions.T - targets) / len(targets)
            self.loss_grad = loss_grad
            # Perform the backpropagation step
            self.back_propagation()
            # (Optional) Print the loss to monitor progress
            if epoch % 10 == 0:  # Print every 10 epochs, adjust as needed
                print(f"Epoch {epoch}, Loss: {mse_loss}")
                self.train_errors.append(mse_loss)

    def see_loss_grad(self):
        '''
        Graph of loss gradient
        '''
        if len(self.loss_grad) > 0:
            plt.plot(self.train_errors)
            plt.title('Loss Gradient')
            plt.show()
            plt.close()

    def pass_data(self, div: int = 1):
        ''' Subprocess 2
                prepares weights for layer transfer
            returns np.array([weights, input])
        '''
        self.weights = [n.weights for n in self.neurons.values()]
        # Pair up the elements from the two lists
        paired_data = list(zip(self.weights, self.input))
        # Randomly select half of the pairs
        n = len(paired_data)
        selected_pairs = random.sample(paired_data, n//div)
        # Separate the pairs back into two lists
        half_weights, input_half = zip(*selected_pairs)
        return half_weights, input_half


if __name__ == "__main__":
    num_points = 100
    test_size = 0.3
    random_state = 42  # For reproducibility

    # Generate dataset
    x = np.linspace(1, 2*np.pi, num_points)
    y = np.sin(x)
    non_lin_test = np.array(list(zip(x, y)))
    input_layer = Layer(layer_size=128, input=non_lin_test)
    input_layer.create_neurons(group_size=3)
    weights_input = weights = input_layer.pass_data(2)[0]
    hidden_layer_one = Layer(
        layer_size=64, input=non_lin_test, weights=weights_input)
    hidden_layer_one.create_neurons(3)
    print(hidden_layer_one.neurons.values())
