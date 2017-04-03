import numpy as np
from data_generator import DataGenerator
import argparse
import initialize_conv_weights
from logger import log
import ctc_loss
from conv_feed_fwd import conv_feed_forward as conv_fwd
from conv_feed_fwd import max_pooling
from sigmoid import sigmoid
import timeit

np.set_printoptions(threshold=np.nan)


class cnn_model(object):
    def __init__(self, epochs):
        self.layer_type = [0, 0, 1, 2]   # 0 for conv layer, 1 for max pooling, 2 for fully connected
        self.epochs = epochs
        # self.context_window = 15
        self.filter_size = [(10, 10), (8, 1), (0, 0), (0, 28)]  # filter size for different conv layers
        self.strides = 1
        self.pooling_size = 2
        self.conv_depth = [1, 1, 1, 1]   # depth of layers - always 1 for FC
        # self.layers = 1    # convolution, max pooling and fully connected are different layers
        self.weights = {}

    def softmax(self, output):
        sf = np.exp(output) / np.sum(np.exp(output))
        return sf

    def feedforward(self, xi, exp_out, MBsize, weights):
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

        num_samples = xi.shape[1]  # num of samples in each file in the mini batch
        num_features = xi.shape[2]  #no.of frequency bands in a sample
        y = []
        y_softmax = []

        #3D array to store combined samples according to the filter size, for all the samples
        samples_combined = np.zeros((
            num_samples - self.filter_size[0] + 1, self.filter_size[0], num_features))   #num_samples - filter_size + 1

        #return
        layers = self.layers
        #conv_layer_pos
        max_pool_layer_pos = [0,1]
        conv_layer = {}
        max_pool_layer = {}
        # calculating gate values of samples
        for file_num in range(MBsize):
            expected_output = str(exp_out[file_num])
            for layer in range(layers):
                #if
                conv_layer[layer] = np.array(conv_fwd(xi, weights, self.filter_size, self.conv_depth))
                #if max_pool_layer[layer] == 1:
                    #max_pool_layer[layer] = max_pooling(conv_layer[layer], self.pooling_size)
                print conv_layer[layer].shape
                return



    def labels(self, probs):
        """
        :param probs:
        :return:
        """
        labels = np.argmax(probs, axis=1)
        return labels

    def cnn_model_train(self, datagen, MBsize):
        """
        :param input_data: sorted input data, by length, in the form of spectrograms
        :param output_data: output transcripts for the audio files
        :param MBsize: size of the mini batch
        :param epochs: no. of iterations on the dataset
        """
        # Calculation of weight size for convolution layer
        size_FC = 161
        for k in range(len(self.layer_type)):
            if self.layer_type[k] == 1:
                if size_FC % self.pooling_size != 0:
                    size_FC = size_FC / self.pooling_size + 1
                else:
                    size_FC /= self.pooling_size
            if self.layer_type[k] == 2:
                self.filter_size[k][0] = size_FC

        epochs = self.epochs
        for i in range(len(self.layer_type)):
            if self.layer_type[i] == 1:
                self.weights[i] = 0
            else:
                self.weights[i] = initialize_conv_weights.init_weights(self.filter_size[i], self.conv_depth[i])
                print self.weights[i].shape

        flag = 0
        for epoch in range(epochs):
            for i, batch in enumerate(datagen.iterate_dev(MBsize, sortBy_duration=True)):
                xi = batch['x']
                y = batch['y']
                exp_out = batch['texts']
                input_lengths = batch['input-lengths']
                label_lengths = batch['label-lengths']
                y, y_softmax, loss = self.feedforward(xi, exp_out, MBsize, weights)
                # print y_softmax.shape
                # log.info(np.array_str(self.labels(y_softmax)))
                # log.info(y)
                flag += 1
                if flag == 5:
                    return
                    # return


def main(dev_desc_file, mini_batch_size, epochs):
    datagen = DataGenerator()
    datagen.load_dev_data(dev_desc_file)
    new_model = cnn_model(epochs)
    trained_cnn_model = new_model.cnn_model_train(datagen, mini_batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_desc_file', type=str,
                        help='Path where JSON file for transcripts and audio files is stored', default='data.json')
    parser.add_argument('--mini_batch_size', type=int, default=1,
                        help='Size of mini batch for training')
    parser.add_argument('--epochs', type=int, default=1, help='No. of epochs to train the model')
    args = parser.parse_args()

    main(args.dev_desc_file, args.mini_batch_size, args.epochs)