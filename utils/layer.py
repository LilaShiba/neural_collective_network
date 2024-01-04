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

    def create_neurons(self, group_size: int = 3, random_activation=False):
        '''
        Initialzes neurons in circular fashion
        2x3s
        '''
        n = len(self.input)
        p_of_x = True  # TODO update to allow for neurons with diverse activations
        if random_activation:
            p_of_x = np.random.choice([0, 1])
        for i in range(n):
            # circular array
            delta_group = [self.input[(i + j) % n] for j in range(group_size)]
            self.neurons[i] = Neuron(
                inputs=delta_group, layer=self.layer, tanh=p_of_x)

    def feed_forward(self):
        '''
        Activate neurons based on their inputs
        '''
        outputs = [neuron.activate(tanh=neuron.tanh)
                   for neuron in self.neurons.values()]
        return outputs

    def back_propagation(self):
        '''

        '''
        pass
