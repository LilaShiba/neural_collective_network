from utils.network import Network
import numpy as np
import random
import matplotlib.pyplot as plt


def sine_wave(graph: bool = False):
    ''' Create data as sine-wave to test for
        non-linearity

        Returns: NP.array sine wave
    '''
    sample_size = 1000
    x_min = 0
    x_max = 2 * np.pi

    # Generate the x values
    x_values = np.linspace(x_min, x_max, sample_size)
    # Compute the y values using a non-linear function (e.g., sine)
    y_values = np.sin(x_values)
    # Combine x and y into a nested list of tuples
    dataset = np.array(list(zip(x_values, y_values)))
    if graph:
        plt.plot(dataset[:, 0], dataset[:, 1])
        plt.title('Input')
        plt.show()
    return dataset


if __name__ == "__main__":

    # Create Sine wave for testing
    non_linearity_test = sine_wave()
    example_network = Network(dataset=non_linearity_test)
    example_network.train(epochs=2)
