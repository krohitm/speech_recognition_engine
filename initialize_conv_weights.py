
import numpy as np
from utils import weights_random


class initialize_weights(object):

    def __init__(self, filter_size, conv_depth):
        self.filter_size = filter_size
        self.conv_depth = conv_depth

    """this function is to initialize convolutional weights
    the weights array is in the form [filter size(as 1 D) * no. of grids or depth of conv layer"""
    def initialize_conv_weights(self):
        FS = self.filter_size
        depth = self.conv_depth
        weights = np.empty([FS**2, depth])
        for i in range(depth):
            weights[:,i] = np.reshape(weights_random(FS**2, 1), FS**2)
        return weights

def main(filter_size, conv_depth):
    new_obj = initialize_weights(filter_size, conv_depth)
    initial_weights = new_obj.initialize_conv_weights()
    return initial_weights

#main(5, 3)