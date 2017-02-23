
# coding: utf-8

# In[1]:

import wave
from scipy.signal import spectrogram
import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy


# In[2]:

data, samplerate = sf.read(
    '/Users/GodSpeed/Documents/Courses/Machine Learning/Project/LibriSpeech/dev-clean/84/121123/84-121123-0004.flac')


# In[8]:

print len(data)
print samplerate
#taking 20ms samples
fft_length = 0.001 * 20 * samplerate
overlap_length = 0.001 * 10 * samplerate
#print window_size
f, t, Sxx = spectrogram(data, fs = samplerate, nperseg = fft_length, 
                        noverlap = overlap_length)
print "No. of sample frequencies: ", len(f) #len(signal.spectrogram(data)[0])
print "No. of time segmnets:", len(t)
print "Spectrogram length: ", len(Sxx)
print Sxx.shape
#data is in the form [frequencies[time]]
#print type(Sxx)
#for i in range(len(t) -  1):
#    print t[i+1] - t[i]


# In[9]:

plt.pcolormesh(t,f,Sxx)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.show()
#print Sxx[:,313]


# ## Output labels

# In[10]:

output_labels = {}
for i in range(26):
    output_labels[i] = chr(i+97)
output_labels[i+1] = ' '
output_labels


# ### Initialize weights
# Following conventions in SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS

# In[11]:

#set layer sizes
layers_sizes = [len(Sxx), 200, len(output_labels)]
input_size = len(Sxx)

"""this function is to initialize wi, 
weights from input layer to first hidden layer,
in the form [input layer[hidden layer]]"""
def initialize_wx(layers_sizes):
    input_size = layers_sizes[0]
    first_hidden_layer_size = layers_sizes[1]
    e = np.sqrt(6)/np.sqrt(input_size + first_hidden_layer_size)
    wx = np.dot(np.random.rand(input_size + 1, 
                               first_hidden_layer_size), 2*e) - e
    return wx

"""this function is to initialize wh,
weights from each hidden layer to next hidden layer
creating wh array as [layer number[input layer[output layer]]]"""
def initialize_W_hh_next(layers_sizes):
    hidden_layers_sizes = layers_sizes[1:-1]
    num_layers = len(hidden_layers_sizes)
    #return empty if only one hidden layer
    if num_layers == 1:
        return
    wh = np.zeros((num_layers - 1, max(hidden_layers_sizes)+1,
                   max(hidden_layers_sizes[1:len(hidden_layers_sizes)])+1))
    for i in range(num_layers - 1):
        e = np.sqrt(6)/np.sqrt(hidden_layers_sizes[i]+hidden_layers_sizes[i+1])
        wh[i, 0:hidden_layers_sizes[i]+1, 0:hidden_layers_sizes[i+1]] = np.dot(
            np.random.rand((hidden_layers_sizes[i]+1), hidden_layers_sizes[i+1]),
            2*e)-e
    return wh

"""this function is to initialize wi, 
weights from last hidden layer to output layer,
in the form [hidden layer[output layer]]"""
def initialize_wo(layers_sizes):
    last_hidden_layer_size = layers_sizes[-2]
    output_size = layers_sizes[-1]
    wo = initialize_wx([layers_sizes[-2],layers_sizes[-1]])
    return wo

"""initialize recurrent weights for hidden layer on itself"""
def initialize_wh(layers_sizes):
    hidden_layers_sizes = layers_sizes[1:-1]
    num_hidden_layers = len(hidden_layers_sizes)
    wh = np.zeros((max(hidden_layers_sizes)+1, num_hidden_layers))
    #print w_hh_curr.shape
    #return
    for i in range(num_hidden_layers):
        e = np.sqrt(6)/np.sqrt(2*hidden_layers_sizes[0])
        wh[0:hidden_layers_sizes[i]+1, i] = np.dot(
            np.random.rand(hidden_layers_sizes[i]+1), 2*e) - e
    return wh

def initialize_wc(layers_sizes):
    wc = initialize_wh(layers_sizes)
    return wc


#initialize weights from input to 1st hidden layer
w_xi = initialize_wx(layers_sizes)

#initialize w_hi
w_hi = initialize_wh(layers_sizes)

#initialize w_ci
w_ci = initialize_wc(layers_sizes)

#initialize w_xf
w_xf = initialize_wx(layers_sizes)

#initialize w_hf
w_hf = initialize_wh(layers_sizes)

#initialize w_cf
w_cf = initialize_wc(layers_sizes)

#initialize w_xc
w_xc = initialize_wx(layers_sizes)

#initialize w_hc
w_hc = initialize_wh(layers_sizes)

#initialize w_xo
w_xo = initialize_wx(layers_sizes)

#initialize w_ho
w_ho = initialize_wh(layers_sizes)

#initialize w_co
w_co = initialize_wc(layers_sizes)


#initialize weights from each hidden layer to next hidden layer
wh_next = initialize_W_hh_next(layers_sizes)

#initialize weights from last hidden layer to output layer
w_hy = initialize_wo(layers_sizes)

#initialize recurrent weights to 0
#w_hh_curr = initialize_w_hh_curr(layers_sizes)

#print w_hi.shape
#print w_xi.shape
#print wh_next.shape
#print w_hy.shape
#print w_hh_curr.shape


# Initialize h(t-1) and c(t-1) to 0

# In[23]:

def initialize_empty_state(layers_sizes):
    hidden_layers_sizes = layers_sizes[1:-1]
    num_hidden_layers = len(hidden_layers_sizes)
    empty_state = np.zeros((max(hidden_layers_sizes)+1, num_hidden_layers))
    return empty_state

hidden_state_prev = initialize_empty_state(layers_sizes)
cell_state_prev = initialize_empty_state(layers_sizes)