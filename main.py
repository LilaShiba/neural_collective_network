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
    dataset = np.array(list(zip(x_values, y_values)))
    # plt.plot(dataset[:, 0], dataset[:, 1])
    # plt.title('Input')
    # plt.show()

    # Input Layer
    input_layer = Layer(input=dataset)
    # Subprocess 1: Create, train, view
    input_layer_weights = input_layer.cycle()
    # Subprocess 2: select and transfer weights, input
    half_weights, input_half = input_layer.pass_data()
    # Hidden Layer: Subprocess 1
    h_one = Layer(input=input_half, weights=half_weights)
    input_layer_weights = h_one.cycle(see_graph=True)
    # Subprocess 2:
    half_weights, input_half = h_one.pass_data()
    out_put_layer = Layer(input=input_half, weights=half_weights)
    # Output Layer: Subprocess
    output_layer_weights = out_put_layer.cycle(see_graph=True)
    # Subprocess 3:
    # Prediction
