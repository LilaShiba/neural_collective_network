from utils.layer import Layer
import numpy as np

if __name__ == "__main__":
    sample_size = 100
    x_min = 0
    x_max = 2 * np.pi
    # Generate the x values
    x_values = np.linspace(x_min, x_max, sample_size)

    # Compute the y values using a non-linear function (e.g., sine)
    y_values = np.sin(x_values)

    # Combine x and y into a nested list of tuples
    dataset = list(zip(x_values, y_values))

    # Print the first few elements to verify
    # print(dataset[:5])
    layer_one = Layer(input=dataset)
    layer_one.create_neurons()
    print('neurons created')
    layer_one.feed_forward()
    print('feed forward successful')
