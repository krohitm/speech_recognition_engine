import numpy as np
from data_generator import DataGenerator
import argparse
import initialize_conv_weights
from scipy import signal
import ctc_loss
from logger import log
from sigmoid import sigmoid

np.set_printoptions(threshold=np.nan)


class cnn_model(object):
    def __init__(self, epochs):
        self.layer_type = [0, 0, 1, 2]   # 0 for conv layer, 1 for max pooling, 2 for fully connected
        self.epochs = epochs
        self.filter_size = [(10, 10), (8, 1), (0, 0), (0, 28)]  # filter size for different conv layers
        self.strides = 1
        self.pooling_size = 2
        self.conv_depth = [1, 1, 1, 1]   # depth of layers - always 1 for FC
        self.weights = {}
        self.bias = {}
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.max_pooling_indices = {}
        self.learning_rate = 0.1
        self.number_of_labels = 28

    def softmax(self, output):
        aa = np.sum(np.exp(output), axis=1)
        sf = np.exp(output) / np.sum(np.exp(output))
        return sf

    def delta_net(self, y):
        return np.multiply(y, np.subtract(np.ones(y.shape), y))

    def feedforward(self, exp_out):
        """
        :param x: input audio signal in the form of spectrogram
        :param exp_out: expected output transcription
        """
        for l in range(len(self.layer_type)):
            if self.layer_type[l] == 0:
                # Feed forward for convolution layer
                self.conv_feed_forward(l, self.layer_inputs[l], self.weights[l],
                                       self.conv_depth[l])
            elif self.layer_type[l] == 1:
                # Feed forward for Max-pooling layer
                self.max_pooling(l, self.layer_inputs[l], self.conv_depth[l])
            else:
                # Feed forward for fully connected layer
                self.fc_feed_forward(l, self.layer_inputs[l], self.weights[l])

    def conv_feed_forward(self, layer_num, x_temp, weights, conv_depth):
        conv_layer = []
        for depth in range(conv_depth):
            conv_frame = signal.convolve2d(x_temp[depth], np.rot90(weights[:, :, depth], 2),
                                           mode='same', boundary='fill', fillvalue=0)
            conv_layer.append(sigmoid(np.add(conv_frame, self.bias[layer_num][:, :, depth])))
        self.layer_outputs[layer_num] = conv_layer
        self.layer_inputs[layer_num+1] = conv_layer

    def max_pooling(self, layer_num, x_temp, conv_depth):
        final_output = []
        final_output_indices = []
        for depth in range(conv_depth):
            temp = x_temp[depth]
            print temp.shape
            temp_list = []
            temp_list_2 = []
            for rows in temp:
                np_zeroes = np.zeros(rows.shape)
                iterator = len(rows) / self.pooling_size + 1
                temp_list.append([np.amax(rows[self.pooling_size * i: self.pooling_size * i + self.pooling_size])
                                    for i in range(iterator)])
                aa = [np.argmax(rows[self.pooling_size * i: self.pooling_size * i + self.pooling_size])
                      + (i * self.pooling_size) for i in range(iterator)]
                for values in aa:
                    np_zeroes[values] = 1
                temp_list_2.append(np_zeroes)
            final_output.append(np.asarray(temp_list).T)
            final_output_indices.append(np.asarray(temp_list_2).T)
        self.layer_outputs[layer_num] = final_output
        self.layer_inputs[layer_num+1] = final_output
        self.max_pooling_indices[layer_num] = final_output_indices

    def fc_feed_forward(self, layer_num, x_temp, weights):
        final_output = np.zeros((x_temp[0].T.shape[0], weights.shape[1]))
        for i in range(len(x_temp)):
            # Adding multiple depth output
            final_output = np.add(final_output, np.dot(x_temp[i].T, weights[:, :, i]))
            final_output = np.add(final_output, self.bias[layer_num][:, :, i])
        self.layer_outputs[layer_num] = self.softmax(sigmoid(final_output))

    def backpropagate(self, loss):
        print "Inside backpropagate"
        for l in range(len(self.layer_type)):
            # Backpropagate from last layer
            layer = len(self.layer_type) - l - 1
            if self.layer_type[layer] == 2:
                # Backpropagate for fully connected layer
                self.backpropagate_FC(layer, self.layer_inputs[layer], loss)

    def backpropagate_FC(self, layer_num, x_temp, loss):
        o = np.zeros((x_temp[0].T.shape[0], self.weights[layer_num][:, :, 0].shape[1]))
        o.fill(loss)
        temp = np.multiply(self.layer_outputs[layer_num], 1.0 - self.layer_outputs[layer_num])
        o = np.multiply(o, temp)
        for i in range(len(x_temp)):
            self.weights[layer_num][:, :, i] = np.add(self.weights[layer_num][:, :, i],
                                                      self.learning_rate*np.dot(x_temp[i], o))
            self.bias[layer_num][:, :, i] = np.add(self.bias[layer_num][:, :, i],
                                                    self.learning_rate * np.dot(np.ones((1, x_temp[i].shape[1])), o))

    def labels(self, probs):
        """
        :param probs:
        :return:
        """
        labels = np.argmax(probs, axis=1)
        return labels

    def cnn_model_train(self, datagen, MBsize):
        """
        :param MBsize: size of the mini batch
        :param datagen: Temp
        """
        # Initialize weights
        self.initialize_network_weights_bias()

        for epoch in range(self.epochs):
            for i, batch in enumerate(datagen.iterate_dev(MBsize, sortBy_duration=True)):
                x = batch['x']
                y = batch['y']
                output = batch['texts']
                for k in range(MBsize):
                    # Prepare input for the first layer
                    expected_output = str(output[k])
                    self.prepare_input_first_layer(x[k, :, :])
                    self.feedforward(expected_output[k])
                    loss = self.calculate_ctc_loss(expected_output)
                    self.backpropagate(loss)
                    self.layer_inputs = {}
                    self.layer_outputs = {}

    def calculate_ctc_loss(self, expected_output):
        ctc_loss_input = np.log(self.layer_outputs[len(self.layer_type) - 1])
        most_probable_output, loss, = ctc_loss.calculate_ctc_loss(ctc_loss_input, e=expected_output)
        log.debug("After CTC Loss calculation :")
        log.debug("Expected output: " + expected_output)
        log.debug("Most probable output: " + str(most_probable_output))
        log.debug("Loss: " + str(loss))
        return loss

    def prepare_input_first_layer(self, x):
        input_layer = []
        for n in range(self.conv_depth[0]):
            input_layer.append(x)
        self.layer_inputs[0] = input_layer

    def initialize_network_weights_bias(self):
        # Calculation of weight size for convolution layer
        size_FC = 161
        for k in range(len(self.layer_type)):
            if self.layer_type[k] == 1:
                if size_FC % self.pooling_size != 0:
                    size_FC = size_FC / self.pooling_size + 1
                else:
                    size_FC /= self.pooling_size
            if self.layer_type[k] == 2:
                self.filter_size[k] = (size_FC, self.number_of_labels)

        num_neurons = 161
        for i in range(len(self.layer_type)):
            if self.layer_type[i] == 1:
                self.weights[i] = 0
                self.bias[i] = 0
            else:
                self.weights[i] = initialize_conv_weights.init_weights(self.filter_size[i], self.conv_depth[i])
                if i == 0:
                    self.bias[i] = initialize_conv_weights.init_bias(num_neurons, self.conv_depth[i])
                else:
                    if self.layer_type[i] == 2:
                        self.bias[i] = initialize_conv_weights.init_bias(self.number_of_labels, self.conv_depth[i])
                    else:
                        if self.layer_type[i-1] == 1:
                            num_neurons = num_neurons/self.pooling_size + 1
                            self.bias[i] = initialize_conv_weights.init_bias(num_neurons, self.conv_depth[i])
                        else:
                            self.bias[i] = initialize_conv_weights.init_bias(num_neurons, self.conv_depth[i])


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
