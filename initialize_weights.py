
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
            intermediate_weights['w_xi'] = np.array(weights_random(LS[i-1]+1, LS[i]))
            intermediate_weights['w_xf'] = weights_random(LS[i-1]+1, LS[i])
            intermediate_weights['w_xc'] = weights_random(LS[i-1]+1, LS[i])
            intermediate_weights['w_xo'] = weights_random(LS[i-1]+1, LS[i])
            intermediate_weights['w_hi'] = weights_random(LS[i], LS[i])
            intermediate_weights['w_hf'] = weights_random(LS[i], LS[i])
            intermediate_weights['w_hc'] = weights_random(LS[i], LS[i])
            intermediate_weights['w_ho'] = weights_random(LS[i], LS[i])
            weights_dict['hidden'+str(i)] = intermediate_weights
        weights_dict['output'] = weights_random(LS[-2]+1, LS[-1])
        return weights_dict

    """this function is to initialize hidden state of hidden layers"""
    def initialize_hidden_state(self):
        LS = self.layers_sizes
        num_layers = len(LS)
        #hidden_layers_sizes = LS[1:-1]
        #num_hidden_layers = len(hidden_layers_sizes)
        hidden_state = {}
        for i in range(1, num_layers - 1):
            hidden_state['hidden'+str(i)] = np.array(weights_random(LS[i], 1))
        return hidden_state


#bias_i = weights_random(1,1)
#bias_f = weights_random(1,1)
#bias_c = weights_random(1,1)
#bais_y = weights_random(1,1)
#bias_o = weights_random(1,1)

def main():
    new_obj = initialize_weights([161, 200, 100, 27])
    initial_weights = new_obj.initialize_weight_dict()
    hidden_state = new_obj.initialize_hidden_state()

main()