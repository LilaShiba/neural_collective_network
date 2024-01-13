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
        '''train all layers in network save for input'''

       # skiping layer 0 as that is the input layer
        for idx in range(epochs):
            delta_layer = Layer(input=self.df, weights=self.delta_weights)
            self.layers[idx] = delta_layer
            self.delta_weights = delta_layer.cycle(see_graph=True)

    def predict(self, test_params: list()) -> np.array:
        '''
        make a prediction on the trained network based on
        a list of testing parameters
        '''
        res = []
        for dp_id, data in enumerate(test_params):
            # print(f'id {dp_id} datapoint {data}')
            for idx, current_layer in enumerate(self.layers.values()):
                predictions = current_layer.feed_forward(data)
                # print(f'layer {idx}')
                # print(' mu:', np.mean(predictions),
                #       'sigma:', np.std(predictions)
                if idx % 2 != 0:
                    res.append(np.mean(predictions))
        return res

    def update_edges(self):
        pass
