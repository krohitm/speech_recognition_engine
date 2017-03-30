import numpy as np

def sigmoid(self, z):
    """z: net input from previous layer
    a: sigmoid activation of net input"""
    a = 1.0 / (1.0 + np.exp(-1.0 * z))
    return a