import numpy as np
#rom scipy import ndimage
from scipy import signal

def conv_feed_forward(xi, weights, filter_size, conv_depth):
    """

    :param xi:
    :param weights:
    :param filter_size:
    :param conv_depth:
    :return:
    """
    #ndimage.convolve(xi, weights, mode='constant', cval=0.0)
    xi = xi.reshape(143,161)
    print xi.shape
    conv_layer = []
    for depth in range(conv_depth):
        weight = np.reshape(weights[:,depth], (filter_size, filter_size))
        conv_frame = signal.convolve2d(xi, np.rot90(weight, 2),
                                   mode = 'same', boundary = 'fill', fillvalue=0)
        conv_layer.append(conv_frame)
    return conv_layer

def max_pooling():

    return