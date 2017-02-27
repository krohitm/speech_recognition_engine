import numpy as np
from data_generator import DataGenerator
import argparse
import initialize_weights

class train_rnn_model(object):

    def __init__(self, input_data, output_data, epochs = 10):
        self.input_data = input_data
        self.output_data = output_data
        self.epochs = epochs

    def sigmoid(z):
        """z: net input from previous layer
        a: sigmoid activation of net input"""
        a = 1.0/(1.0+np.exp(-z))
        return a
    
    def tanh(z):
        """z: net input from previous layer
        a: tanh activation of input"""
        a = (np.exp(-z) - np.exp(-z))/(np.exp(-z) + np.exp(-z))
        return a
    
    def feedforward(xi):
        """mini_batch: mini batch for training is to be done
        return: input_gate: input gate values for the mini batch
                forget_gate: forget gate values for the mini batch
                cell_state_gate: cell state for the mini batch
                output_gate: output gate values for the mini batch
                hidden_state: hidden state values for the mini batch
                y: output"""
        input_gate = sigmoid(np.dot(np.transpose(w_xi), xi)
                             + (w_hi * hidden_state_prev) 
                             + (w_ci * cell_state_prev) + bias_i)
        forget_gate = sigmoid(np.dot(np.transpose(w_xf), xi) + 
                              (w_hf * hidden_state_prev) + 
                              (w_cf * cell_state_prev) + bias_f)
        cell_state_gate = (forget_gate * cell_state_prev) + (input_gate * tanh(np.dot(
                    np.transpose(w_xc), xi) + (w_hc * hidden_state_prev) + bias_c))
        output_gate = sigmoid(np.dot(np.transpose(w_xo), xi) 
                              + (w_ho * hidden_state_prev) 
                              + (w_co * cell_state_gate) + bias_o)
        hidden_state = output_gate * tanh(cell_state_gate)
        y = np.dot(np.transpose(w_hy), hidden_state) + bias_y
        return input_gate, forget_gate, cell_state_gate, output_gate, hidden_state, y
    
    #def CTC(y):
        
    
    def rnn_lstm_model_train(self, datagen, epochs = 1):
        """input_data: sorted input data, by length, in the form of spectrograms
        output_data: output transcripts for the audio files
        mini_batch_size: size of the mini batch
        epochs: no. of iterations on the dataset"""
        for epoch in epochs:
            for i, batch in enumerate(datagen.iterate_dev(50, sortBy_duration = True)):
                xi = batch['x']
                input_gate, forget_gate, cell_state_gate, output_gate, hidden_state, y = \
                    feedforward(xi, )

def main(dev_desc_file, epochs):
    datagen = DataGenerator()
    datagen.load_dev_data(dev_desc_file)

    trained_rnn_model = train_rnn_model.rnn_lstm_model_train(datagen, epochs = 10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dev_desc_file', type = str,
        help = 'Path where JSON file for transcripts and audio files is stored')
    parser.add_argument('--epochs', type = int, default = 10,
        help = 'No. of epochs to train the model')
    args = parser.parse_args()

    main(args.dev_desc_file, args.epochs)

