from utils.layer import Layer
import numpy as np
import matplotlib.pyplot as plt
from typing import *


class Network:
    '''
    Network of 3x2 Neurons
    '''

    def __init__(self, dataset: np.array):
        self.df = dataset
        self.input_layer = Layer(input=dataset)
        self.layers = dict()
        self.layers[0] = self.input_layer
        self.delta_weights = self.input_layer.cycle()

    def train(self, epochs: int = 5):

       # skiping layer 0 as that is the input layer
        for idx in range(epochs):
            delta_layer = Layer(input=self.df, weights=self.delta_weights)
            self.layers[idx] = delta_layer
            self.delta_weights = delta_layer.cycle(see_graph=True)
