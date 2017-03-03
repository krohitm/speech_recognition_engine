import numpy as np
from data_generator import DataGenerator
import argparse
import initialize_weights

class train_rnn_model(object):

    def __init__(self, epochs = 10):
        #self.input_data = input_data
        #self.output_data = output_data
        self.epochs = epochs

    def sigmoid(self, z):
        """z: net input from previous layer
        a: sigmoid activation of net input"""
        a = 1.0/(1.0+np.exp(-z))
        return a
    
    def tanh(self, z):
        """z: net input from previous layer
        a: tanh activation of input"""
        a = (np.exp(-z) - np.exp(-z))/(np.exp(-z) + np.exp(-z))
        return a
    
    def feedforward(self, xi, MBsize, weights, hidden_states_prev):
        """
        :param xi: input audio signals batch in the form of spectrogram
        :param weights: weights of the layers in a dictionary
        :param hidden_states: hidden states of the hidden layers
        :param MBsize: mini batch for training is to be done
        :return input_gate: input gate values for the mini batch
        :return forget_gate: forget gate values for the mini batch
        :return cell_state_gate: cell state for the mini batch
        :return output_gate: output gate values for the mini batch
        :return hidden_state: hidden state values for the mini batch
        :return y: output
        """

        num_hidden_layers = len(hidden_states_prev)  #num of hidden layers

        num_samples = xi.shape[1]   #num of samples in each file in the mini batch


        #gates are dictionary of the form {hidden_layer_number[samples*neurons in hidden layer]}
        input_gate = {}
        forget_gate = {}
        cell_state_gate = {}
        output_gate = {}
        hidden_state = {}
        y = []
        dict_check ={}

        hidden_layers = sorted(hidden_states_prev.keys())
        #print hidden_states_prev[1].shape

        for layer in hidden_layers:
            temp_array = np.zeros((num_samples, hidden_states_prev[layer].shape[0]))
            input_gate[layer] = forget_gate[layer] = cell_state_gate[layer] = \
                output_gate[layer] = hidden_state[layer] = temp_array


        for file_num in range(MBsize):
            for sample_num in range(num_samples):
                input = np.append(xi[file_num,sample_num,:], 1) #taking one sample at a time
                #print input
                #print input.shape
                #print xi.shape[1]
                #print xi.shape[2]
                for layer in hidden_layers: #need hidden layers in sorted order
                    #print input_gate[layer][sample_num].shape
                    #print np.squeeze(
                    #    np.expand_dims(np.dot(np.transpose(weights[layer]['w_xi']), input), axis = 1) +
                    #    np.dot(np.transpose(weights[layer]['w_hi']), hidden_states_prev[layer])).shape
                    #return

                    input_gate[layer][sample_num] = np.squeeze(self.sigmoid(
                        np.expand_dims(np.dot(np.transpose(weights[layer]['w_xi']), input), axis = 1) +
                        np.dot(np.transpose(weights[layer]['w_hi']), hidden_states_prev[layer])))
                    #+ bias_i)
                    print input_gate[layer][sample_num].shape
                    return
        #remove previous cell state from all gate equations
        forget_gate = self.sigmoid(np.dot(np.transpose(w_xf), xi) +
                              np.dot(np.transpose(w_hf), hidden_state_prev) +
                              (w_cf * cell_state_prev) + bias_f)
        cell_state_gate = (forget_gate * cell_state_prev) + (input_gate * tanh(np.dot(
                    np.transpose(w_xc), xi) + np.dot(np.transpose(w_hc), hidden_state_prev)
                                                                               + bias_c))
        output_gate = self.sigmoid(np.dot(np.transpose(w_xo), xi)
                              + np.dot(np.transpose(w_ho), hidden_state_prev)
                              + (w_co * cell_state_gate) + bias_o)
        hidden_state = output_gate * self.tanh(cell_state_gate)
        y = np.dot(np.transpose(w_hy), hidden_state) + bias_y
        return input_gate, forget_gate, cell_state_gate, output_gate, hidden_state, y
    
    #def CTC(y):
        
    
    def rnn_lstm_model_train(self, datagen, MBsize, epochs):
        """
        :param input_data: sorted input data, by length, in the form of spectrograms
        :param output_data: output transcripts for the audio files
        :param MBsize: size of the mini batch
        :param epochs: no. of iterations on the dataset
        """

        weights, hidden_states = initialize_weights.main()
        #print hidden_states.keys()
        #print hidden_states['hidden2'].shape
        #return
        flag = 0
        for epoch in range(epochs):
            for i, batch in enumerate(datagen.iterate_dev(MBsize, sortBy_duration = True)):
                xi = batch['x']
                y = batch['y']
                text = batch['texts']
                input_lengths = batch['input-lengths']
                label_lengths = batch['label-lengths']
                #print xi.shape
                #print label_lengths
                flag = flag +1
                if flag >= 2:
                    return
                #input_gate, forget_gate, cell_state_gate, output_gate, hidden_state, y = \
                self.feedforward(xi, MBsize, weights, hidden_states)

def main(dev_desc_file, mini_batch_size, epochs):
    datagen = DataGenerator()
    datagen.load_dev_data(dev_desc_file)
    model = train_rnn_model()
    trained_rnn_model = model.rnn_lstm_model_train(
        datagen, mini_batch_size, epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dev_desc_file', type = str,
        help = 'Path where JSON file for transcripts and audio files is stored')
    parser.add_argument('--mini_batch_size', type = int, default = 50,
                        help = 'Size of mini batch for training')
    parser.add_argument('--epochs', type = int, default = 10,
        help = 'No. of epochs to train the model')
    args = parser.parse_args()

    main(args.dev_desc_file, args.mini_batch_size, args.epochs)

