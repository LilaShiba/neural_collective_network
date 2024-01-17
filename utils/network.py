from utils.layers import Layer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import *


class Network:
    '''
    Network of 3x2 Neurons
    '''

    def __init__(self, dataset: np.array):
        self.df = dataset
        n = len(dataset)
        self.input_layer = Layer(input=dataset, layer_size=n)
        self.input_layer.create_neurons(group_size=1)
        self.layers = dict()
        self.layers[0] = self.input_layer
        self.delta_weights = [
            n.weights for n in self.input_layer.neurons.values()]

    def init_network(self, layers: int = 5):
        '''train all layers in network save for input'''

       # skiping layer 0 as that is the input layer
        for idx in range(layers):
            n = len(self.delta_weights)
            delta_layer = Layer(layer_size=n, input=self.df,
                                weights=self.delta_weights)
            self.layers[idx] = delta_layer
            self.delta_weights = delta_layer.create_neurons()
        return self.layers

    def train_network(self, epochs: int = 100):
        '''updates all neurons in place'''
        for idx in range(epochs):
            for layer in self.layers.values():
                layer.iterate()

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

                res.append(np.mean(predictions))
        return res

    def update_edges(self):
        pass

    @staticmethod
    def sine_wave(graph: bool = False):
        ''' Create data as sine-wave to test for
            non-linearity

            Returns: NP.array sine wave
        '''
        sample_size = 128
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

    @staticmethod
    def transduce_img(img_path, output_path=None):
        '''returns a 128x128 grey scale matrix'''
        with Image.open(img_path) as img:
            # Convert the image to grayscale
            grayscale_img = img.convert("L")
            # Resize the image to 128x128
            resized_img = grayscale_img.resize((128, 128))
            # Save the converted image
            image_array = np.array(resized_img)
            if output_path != None:
                resized_img.save(output_path)
        return image_array

    @staticmethod
    def show_image(image_array):
        '''show image from array'''
        # Display the image
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')  # Turn off axis numbers
        plt.show()
