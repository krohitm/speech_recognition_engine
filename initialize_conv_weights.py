
import numpy as np
from utils import weights_random


"""this function is to initialize convolutional weights
    the weights array is in the form [filter size(as 1 D) * no. of grids or depth of conv layer"""


def init_weights(filter_size, conv_depth):
    depth = conv_depth
    weights = np.empty([filter_size[0], filter_size[1], depth])
    for i in range(depth):
        weights[:, :, i] = weights_random(filter_size[0], filter_size[1])
    return weights


def init_bias(number_of_neurons, conv_depth):
    depth = conv_depth
    bias = np.empty((1, number_of_neurons, conv_depth))
    for i in range(depth):
        bias[:, :, i] = weights_random(1, number_of_neurons)
    return bias
