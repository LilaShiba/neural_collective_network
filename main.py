from utils.layer import Layer
import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create data as sine-wave to test for
    # non-linearity
    sample_size = 1000
    x_min = 0
    x_max = 2 * np.pi

    # Generate the x values
    x_values = np.linspace(x_min, x_max, sample_size)
    # Compute the y values using a non-linear function (e.g., sine)
    y_values = np.sin(x_values)
    # Combine x and y into a nested list of tuples
    dataset = list(zip(x_values, y_values))
    plt.plot(dataset)
    plt.title('Input')
    plt.show()

    # Subprocess 1: Create, train, view
    # Input Layer
    input_layer = Layer(input=dataset)
    input_layer.create_neurons()
    input_layer.train()
    input_layer.see_loss_grad()

    # Subprocess 2:
    # extract weights and input for neuron transfer
    input_layer_weights = [n.weights for n in input_layer.neurons.values()]
    # Hidden Layer
    n = len(input_layer_weights)//2
    # Pair up the elements from the two lists
    paired_data = list(zip(input_layer_weights, dataset))
    # Randomly select half of the pairs
    selected_pairs = random.sample(paired_data, n // 2)
    # Separate the pairs back into two lists
    half_weights, input_half = zip(*selected_pairs)

    # Subprocess 1:
    h_one = Layer(input=input_half, weights=half_weights)
    h_one.create_neurons()
    h_one.train()
    h_one.see_loss_grad()

    # Subprocess 2:
    input_layer_weights = [n.weights for n in h_one.neurons.values()]
    # Hidden Layer
    n = len(input_layer_weights)//2
    # Pair up the elements from the two lists
    paired_data = list(zip(input_layer_weights, dataset))
    # Randomly select half of the pairs
    selected_pairs = random.sample(paired_data, n // 2)
    # Separate the pairs back into two lists
    half_weights, input_half = zip(*selected_pairs)
    out_put_layer = Layer(input=input_half, weights=half_weights)

    # Subprocess 1:
    out_put_layer.create_neurons()
    out_put_layer.train()
    out_put_layer.see_loss_grad()

    # Subprocess 3:
    # Prediction
