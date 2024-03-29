from vectors import Vector
from network import Network
from layers import Layer
from neurons import Neuron
import unittest
import numpy as np
import sys
import os
np.set_printoptions(precision=3, suppress=True)


class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.inputs = np.array([0.5, 0.8])
        self.neuron = Neuron(inputs=self.inputs, layer=1)

    def test_initialization(self):
        self.assertEqual(self.neuron.layer, 1)
        self.assertEqual(self.neuron.x, 0.5)
        self.assertEqual(self.neuron.y, 0.8)
        self.assertIsNotNone(self.neuron.weights)
        self.assertIsNotNone(self.neuron.bias)

    def test_iterate(self):
        initial_output = self.neuron.output
        self.neuron.iterate()
        self.neuron.iterate()
        print(f'prediction: {self.neuron.output} res: {self.neuron.y}')
        self.assertNotEqual(initial_output, self.neuron.output)


class TestLayer(unittest.TestCase):
    def setUp(self):
        sine_wave = np.array(Vector.generate_noisy_sin())
        self.layer = Layer(data_points=sine_wave, number_of_neurons=2)

    def test_create_neurons(self):
        inputs, weights = self.layer.create_neurons(2)
        print("")
        print(f"Layer create_neurons inputs")
        print(inputs)
        print("")
        print("weights:")
        print(weights)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(weights), 2)

    def test_feed_forward(self):
        self.layer.create_neurons(2)
        state, output = self.layer.feed_forward()
        print("Layer feed_forward state:")
        print(state)
        print('Layer Output:')
        print(output)
        self.assertIsNotNone(state)
        self.assertIsNotNone(output)


class TestNetwork(unittest.TestCase):
    def setUp(self):
        sine_wave = np.array(Vector.generate_noisy_sin())
        self.network = Network(dataset=sine_wave)

    def test_init_network(self):
        self.network.init_network(layers=3)
        print(f"Network init_network layers: {len(self.network.layers)}")
        self.assertEqual(len(self.network.layers), 4)

    def test_train_network(self):
        self.network.init_network(layers=3)
        initial_output = self.network.layers[0].neurons[0].output
        self.network.train_network(epochs=1)

        self.assertNotEqual(
            initial_output, self.network.layers[0].neurons[0].output)

    def test_predict(self):
        self.network.init_network(layers=2)
        self.network.train_network(epochs=2)
        test_params = [0.5, 0.8]
        predictions = self.network.predict(test_params=test_params)
        # print("")
        # print(
        #     f"Network predict test_params: {test_params}, predictions: {predictions}")
        # print('')
        expected_predictions_length = len(
            test_params) * len(self.network.layers)
        self.assertEqual(len(predictions), expected_predictions_length)


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    unittest.main()
