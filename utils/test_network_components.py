from vectors import Vector
from network import Network
from layers import Layer
from neurons import Neuron
import unittest
import numpy as np
import sys
import os

# Adjust the path to include the directory above 'unit_tests' to access 'utils'
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'utils')))


class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.inputs = np.array([0.5, -0.5])
        self.neuron = Neuron(inputs=self.inputs, layer=1)

    def test_initialization(self):
        self.assertEqual(self.neuron.layer, 1)
        self.assertEqual(self.neuron.x, 0.5)
        self.assertEqual(self.neuron.y, -0.5)
        self.assertIsNotNone(self.neuron.weights)
        self.assertIsNotNone(self.neuron.bias)

    def test_feed_forward(self):
        output = self.neuron.feed_forward()
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), 2)

    def test_iterate(self):
        initial_output = self.neuron.output
        self.neuron.iterate()
        self.assertNotEqual(initial_output, self.neuron.output)


class TestLayer(unittest.TestCase):
    def setUp(self):
        sine_wave = np.array(Vector.generate_noisy_sin())
        self.layer = Layer(data_points=sine_wave, number_of_neurons=2)

    def test_create_neurons(self):
        inputs, weights = self.layer.create_neurons(2)
        # Expecting the number of neurons (and thus inputs and weights) to match the smaller number due to data size
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(weights), 2)

    def test_feed_forward(self):
        self.layer.create_neurons(2)
        state, output = self.layer.feed_forward()
        self.assertIsNotNone(state)
        self.assertIsNotNone(output)


class TestNetwork(unittest.TestCase):
    def setUp(self):
        sine_wave = np.array(Vector.generate_noisy_sin())
        self.network = Network(dataset=sine_wave)

    def test_init_network(self):
        self.network.init_network(layers=3)
        # Expect 4 layers in total: 1 input layer + 3 hidden layers
        self.assertEqual(len(self.network.layers), 4)

    def test_train_network(self):
        self.network.init_network(layers=3)
        initial_output = self.network.layers[0].neurons[0].output
        self.network.train_network(epochs=1)
        self.assertNotEqual(
            initial_output, self.network.layers[0].neurons[0].output)

    def test_predict(self):
        self.network.init_network(layers=3)
        self.network.train_network(epochs=1)
        test_params = [0.5, -0.5]
        predictions = self.network.predict(test_params=test_params)
        # Adjusting the expectation based on the actual implementation details of your predict method
        # Assuming each layer contributes to the prediction, adjust the expected length accordingly
        expected_predictions_length = len(
            test_params) * len(self.network.layers)
        self.assertEqual(len(predictions), expected_predictions_length)


if __name__ == '__main__':
    unittest.main()
