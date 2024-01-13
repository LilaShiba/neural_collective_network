from utils.network import Network
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def sine_wave(graph: bool = False):
    ''' Create data as sine-wave to test for
        non-linearity

        Returns: NP.array sine wave
    '''
    sample_size = 100
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
    # Parameters
    num_points = 100
    test_size = 0.3
    random_state = 42  # For reproducibility

    # Generate dataset
    x = np.linspace(1, 100, num_points)
    # y = 2x + 3 with some noise
    y = np.sin(x)

    # Split dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    training_data = np.array(list(zip(x_train, y_train)))
    # Create Sine wave for testing
    # non_linearity_test = sine_wave()
    # plt.plot(non_linearity_test[:, -1])
    # plt.title('Input')
    # plt.show()
    example_network = Network(dataset=training_data)
    example_network.train(epochs=2)
    res = example_network.predict(test_params=x_test)
    print(np.mean(res-y_test))
