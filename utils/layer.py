import random
import numpy as np
from typing import Tuple, List, Dict
from utils.neuron import Neuron


class Layer:
    '''
    Class representing the temporal steps & connections between complex neurons

    '''

    def __init__(self, input=np.array(np.array)) -> None:
        self.input = input
        self.layer: int = 0
        self.neurons: dict = dict()

    def create_neurons(self, group_size: int = 3):
        '''
        Initialzes neurons in circular fashion
        2x3
        '''
        n = len(self.input)
        for i in range(n):
            # circular array
            delta_group = [self.input[(i + j) % n] for j in range(group_size)]
            self.neurons[i] = Neuron(inputs=delta_group, layer=self.layer)

    def feed_forward(self):
        '''

        '''
        outputs = [neuron.activate() for neuron in self.neurons.values()]
        return outputs

    def back_propagation(self):
        '''

        '''
        pass
