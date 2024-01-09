from utils.layer import Layer
import numpy as np

if __name__ == "__main__":
    sample_size = 1000
    x_min = 0
    x_max = 2 * np.pi

    # Generate the x values
    x_values = np.linspace(x_min, x_max, sample_size)

    # Compute the y values using a non-linear function (e.g., sine)
    y_values = np.sin(x_values)

    # Combine x and y into a nested list of tuples
    dataset = list(zip(x_values, y_values))
    # Input Layer
    input_layer = Layer(input=dataset)
    input_layer.create_neurons()
    input_layer.train()
    input_layer.see_loss_grad()
    input_layer_weights = [n.weights for n in input_layer.neurons.values()]
    # Hidden Layer
    h_one = Layer(input=dataset, weights=input_layer_weights)
    h_one.create_neurons()
    h_one.train()
    h_one.see_loss_grad()
