
import numpy as np
from utils import weights_random


class initialize_weights(object):

    def __init__(self, layers_sizes):
        self.layers_sizes = layers_sizes

    """this function is to initialize weights,
    weights from current layer to next  layer,
    in the form [current layer[next layer]]"""
    def initialize_weight_dict(self):
        LS = self.layers_sizes
        num_layers = len(LS)
        weights_dict = {}
        for i in range(1,num_layers - 1):
            intermediate_weights = {}
            intermediate_weights['w_xi'] = np.array(weights_random(LS[i-1], LS[i]))
            # print intermediate_weights['w_xi'].shape
            intermediate_weights['w_xf'] = weights_random(LS[i-1], LS[i])
            intermediate_weights['w_xc'] = weights_random(LS[i-1], LS[i])
            intermediate_weights['w_xo'] = weights_random(LS[i-1], LS[i])
            intermediate_weights['w_hi'] = weights_random(LS[i], LS[i])
            intermediate_weights['w_hf'] = weights_random(LS[i], LS[i])
            intermediate_weights['w_hc'] = weights_random(LS[i], LS[i])
            intermediate_weights['w_ho'] = weights_random(LS[i], LS[i])
            weights_dict[i] = intermediate_weights
        weights_dict['output'] = weights_random(LS[-2], LS[-1])
        return weights_dict

    """this function is to initialize hidden state of hidden layers"""
    def initialize_hidden_state(self):
        LS = self.layers_sizes
        num_layers = len(LS)
        hidden_states = {}
        cell_states = {}
        for i in range(1, num_layers - 1):
            hidden_states[i] = np.array(weights_random(LS[i], 1))

        for i in range(1, num_layers - 1):
            cell_states[i] = np.array(weights_random(LS[i], 1))

        return hidden_states, cell_states



    def initialize_bias(self):
        LS = self.layers_sizes
        num_layers = len(LS)
        bias = {}

        for i in range(1, num_layers - 1):
            intermediate_bias = {}
            intermediate_bias['i']  = np.array(weights_random(LS[i], 1))
            intermediate_bias['f'] = np.array(weights_random(LS[i], 1))
            intermediate_bias['c'] = np.array(weights_random(LS[i], 1))
            intermediate_bias['y'] = np.array(weights_random(LS[i], 1))
            intermediate_bias['o'] = np.array(weights_random(LS[i], 1))
            bias[i] = intermediate_bias
            bias['y'] = np.array(weights_random(LS[-1], 1))
        return bias


def main():
    new_obj = initialize_weights([161, 200, 100, 29])
    initial_weights = new_obj.initialize_weight_dict()
    hidden_states, cell_states = new_obj.initialize_hidden_state()
    bias = new_obj.initialize_bias()
    return initial_weights, hidden_states, cell_states, bias
