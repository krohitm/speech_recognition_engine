import numpy as np
from data_generator import DataGenerator
import argparse
import initialize_weights
from logger import log
import ctc_loss
import timeit

np.set_printoptions(threshold=np.nan)


class train_rnn_model(object):

    def __init__(self, epochs):
        self.epochs = epochs

    def gate(self, x, w, hidden_state_prev, bias, gate_type):
        """
        :param x: input sample to the layer
        :param w: weight matrix for the hidden layer
        :param hidden_state_prev: previous hidden states of the hidden layer
        :param bias: bias as per the gate for which values are being calculated
        :param gate_type: defines gate type (hidden, output, forget)
        :return: gate_output: gate vector for the hidden layer
        """
        key_x = 'w_x' + gate_type
        key_h = 'w_h' + gate_type

        if gate_type == 'c':
            gate_output = self.tanh(np.dot(np.transpose(w[key_x]), x) +
                                    np.squeeze(np.dot(np.transpose(w[key_h]), hidden_state_prev) +bias[gate_type]))
        else:
            gate_output = self.sigmoid(np.dot(np.transpose(w[key_x]), x) +
                                       np.squeeze(np.dot(np.transpose(w[key_h]), hidden_state_prev) +bias[gate_type]))
        return gate_output

    def cell_state(self, FG, cell_states_prev, input_gate, input, weights, hidden_states_prev, bias):
        gate_output = (FG * np.squeeze(cell_states_prev)) + ( input_gate * np.squeeze(self.gate(input, weights,
                                                                                  hidden_states_prev, bias, 'c')))
        return gate_output

    def sigmoid(self, z):
        """z: net input from previous layer
        a: sigmoid activation of net input"""
        a = 1.0/(1.0+np.exp(-1.0 * z))
        return a
    
    def tanh(self, z):
        """z: net input from previous layer
        a: tanh activation of input"""
        #num = np.exp(-z) - np.exp(-z)
        #den = np.exp(-z) + np.exp(-z)
        a = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
        # log.info(a)
        return a

    def softmax(self, output):
        sf = np.exp(output)/np.sum(np.exp(output))
        return sf
    
    def feedforward(self, xi, exp_out,  MBsize, weights, hidden_states_prev, cell_states_prev, bias):
        """
        :param xi: input audio signals batch in the form of spectrogram
        :param exp_out: expected output transcription
        :param weights: weights of the layers in a dictionary
        :param hidden_states_prev: previous hidden states of the hidden layers
        :param cell_states_prev: previous cell states of the hidden layers
        :param MBsize: mini batch for training is to be done
        :return input_gate: input gate values for the mini batch
        :return forget_gate: forget gate values for the mini batch
        :return cell_state_gate: cell state for the mini batch
        :return output_gate: output gate values for the mini batch
        :return hidden_state: hidden state values for the mini batch
        :return y: output
        """

        # num_hidden_layers = len(hidden_states_prev)  #num of hidden layers

        num_samples = xi.shape[1]   # num of samples in each file in the mini batch

        # gates are dictionary of the form {hidden_layer_number[samples*neurons in hidden layer]}
        input_gate = {}
        forget_gate = {}
        cell_state_gate = {}
        output_gate = {}
        hidden_state = {}
        y = []
        y_softmax = []
        # dict_check ={}

        hidden_layers = sorted(hidden_states_prev.keys())
        # print hidden_states_prev[1].shape

        for layer in hidden_layers:
            # initializing gates for hidden layers
            input_gate[layer] = np.zeros((num_samples, hidden_states_prev[layer].shape[0]))
            forget_gate[layer] = np.zeros((num_samples, hidden_states_prev[layer].shape[0]))
            cell_state_gate[layer] = np.zeros((num_samples, hidden_states_prev[layer].shape[0]))
            output_gate[layer] = np.zeros((num_samples, hidden_states_prev[layer].shape[0]))
            hidden_state[layer] = np.zeros((num_samples, hidden_states_prev[layer].shape[0]))

        # calculating gate values of samples
        for file_num in range(MBsize):
            expected_output = str(exp_out[file_num])
            print expected_output
            for sample_num in range(num_samples):
                # taking one sample at a time
                input = xi[file_num, sample_num,:]
                for layer in hidden_layers:
                    # x, wx, wh, hidden_state_prev, bias
                    input_gate[layer][sample_num] = self.gate(input, weights[layer], hidden_states_prev[layer],
                                                              bias[layer], 'i')
                    forget_gate[layer][sample_num] = self.gate(input, weights[layer], hidden_states_prev[layer],
                                                               bias[layer], 'f')
                    cell_state_gate[layer][sample_num] = self.cell_state(
                        forget_gate[layer][sample_num], cell_states_prev[layer], input_gate[layer][sample_num] ,
                        input, weights[layer], hidden_states_prev[layer], bias[layer])
                    output_gate[layer][sample_num] = self.gate(input, weights[layer], hidden_states_prev[layer],
                                                               bias[layer], 'o')
                    hidden_state[layer][sample_num] = output_gate[layer][sample_num] *\
                                                      self.tanh(cell_state_gate[layer][sample_num])
                    #log.info(hidden_state[layer][sample_num])
                    # input for next layer
                    input = hidden_state[layer][sample_num]
                    hidden_states_prev[layer] = np.expand_dims(hidden_state[layer][sample_num], axis = 1)
                    cell_states_prev[layer] = np.expand_dims(cell_state_gate[layer][sample_num], axis = 1)
                    #print hidden_states_prev[layer].shape
                    #print cell_states_prev[layer].shape
                    #return


                output = np.dot(np.transpose(weights['output']), np.expand_dims(input, axis = 1)) + bias['y']
                #log.info(output)
                y_softmax.append(self.softmax(output))
                y.append(output)

        y = np.array(y)
        y_softmax = np.array(y_softmax)
        # Sample code starts
        # Calculation of CTC Loss
        # Must reshape the output
        aaa = y_softmax#.reshape((143, 29))
        # Take log of the output
        eee = np.log(aaa)
        most_probable_output, loss,  = ctc_loss.calculate_ctc_loss(eee, e=expected_output)
        log.debug("After CTC Loss calculation :")
        log.debug("Most probable output: " + str(most_probable_output))
        log.debug("Loss: " + str(loss))
        # Sample code ends
        #log.info(y_softmax)
        # return
        return input_gate, forget_gate, cell_state_gate, output_gate, hidden_state, y, y_softmax, loss

    def labels(self, probs):
        """

        :param probs:
        :return:
        """
        labels = np.argmax(probs, axis = 1)
        return labels
    
    def rnn_lstm_model_train(self, datagen, MBsize, epochs = 1):
        """
        :param input_data: sorted input data, by length, in the form of spectrograms
        :param output_data: output transcripts for the audio files
        :param MBsize: size of the mini batch
        :param epochs: no. of iterations on the dataset
        """

        weights, hidden_states_prev, cell_states_prev, bias = initialize_weights.main()
        flag = 0
        for epoch in range(epochs):
            for i, batch in enumerate(datagen.iterate_dev(MBsize, sortBy_duration = True)):
                xi = batch['x']
                y = batch['y']
                exp_out = batch['texts']
                input_lengths = batch['input-lengths']
                label_lengths = batch['label-lengths']
                input_gate, forget_gate, cell_state_gate, output_gate, hidden_state, y, y_softmax, loss =\
                    self.feedforward(xi, exp_out, MBsize, weights, hidden_states_prev, cell_states_prev, bias)
                # print y_softmax.shape
                #log.info(np.array_str(self.labels(y_softmax)))
                #log.info(y)
                flag += 1
                if flag == 5:
                    return
                # return


def main(dev_desc_file, mini_batch_size, epochs):
    datagen = DataGenerator()
    datagen.load_dev_data(dev_desc_file)
    model = train_rnn_model(epochs)
    trained_rnn_model = model.rnn_lstm_model_train(datagen, mini_batch_size, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_desc_file', type=str,
                        help='Path where JSON file for transcripts and audio files is stored', default = 'data.json')
    parser.add_argument('--mini_batch_size', type=int, default=1,
                        help='Size of mini batch for training')
    parser.add_argument('--epochs', type=int, default=1, help='No. of epochs to train the model')
    args = parser.parse_args()

    main(args.dev_desc_file, args.mini_batch_size, args.epochs)