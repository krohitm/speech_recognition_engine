import numpy as np
from data_generator import DataGenerator
import argparse
import initialize_conv_weights
from scipy import signal

np.set_printoptions(threshold=np.nan)


class cnn_model(object):
    def __init__(self, epochs):
        self.layer_type = [0, 0, 1, 2]   # 0 for conv layer, 1 for max pooling, 2 for fully connected
        self.epochs = epochs
        self.filter_size = [(10, 10), (8, 1), (0, 0), (0, 28)]  # filter size for different conv layers
        self.strides = 1
        self.pooling_size = 3
        self.conv_depth = [1, 1, 1, 1]   # depth of layers - always 1 for FC
        self.weights = {}
        self.layer_inputs = {}
        self.layer_outputs = {}

    def softmax(self, output):
        sf = np.exp(output) / np.sum(np.exp(output))
        return sf

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
                self.fc_feed_forward(l, self.layer_inputs[l], self.weights[l][:, :, 0])

    def conv_feed_forward(self, layer_num, x_temp, weights, conv_depth):
        conv_layer = []
        for depth in range(conv_depth):
            conv_frame = signal.convolve2d(x_temp[depth], np.rot90(weights[:, :, depth], 2),
                                           mode='same', boundary='fill', fillvalue=0)
            conv_layer.append(conv_frame)
        self.layer_outputs[layer_num] = conv_layer
        self.layer_inputs[layer_num+1] = conv_layer

    def max_pooling(self, layer_num, x_temp, conv_depth):
        final_output = []
        for depth in range(conv_depth):
            temp = x_temp[depth]
            print temp.shape
            temp_list = []
            for rows in temp:
                iterator = len(rows) / self.pooling_size + 1
                temp_list.append([np.amax(rows[self.pooling_size * i: self.pooling_size * i + self.pooling_size])
                                    for i in range(iterator)])
            final_output.append(np.asarray(temp_list).T)
        self.layer_outputs[layer_num] = final_output
        self.layer_inputs[layer_num+1] = final_output

    def fc_feed_forward(self, layer_num, x_temp, weights):
        final_output = np.zeros((x_temp[0].T.shape[0], weights.shape[1]))
        for i in range(len(x_temp)):
            final_output = np.add(final_output, np.dot(x_temp[i].T, weights))
        self.layer_outputs[layer_num] = final_output

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
        self.initialize_network_weights()

        for epoch in range(self.epochs):
            for i, batch in enumerate(datagen.iterate_dev(MBsize, sortBy_duration=True)):
                x = batch['x']
                y = batch['y']
                expected_output = batch['texts']
                for k in range(MBsize):
                    # Prepare input for the first layer
                    self.prepare_input_first_layer(x[k, :, :])
                    self.feedforward(expected_output[k])

        for epoch in range(self.epochs):
            for i, batch in enumerate(datagen.iterate_dev(MBsize, sortBy_duration=True)):
                x = batch['x']
                y = batch['y']
                exp_out = batch['texts']
                input_lengths = batch['input-lengths']
                label_lengths = batch['label-lengths']
                self.feedforward(x, exp_out, self.weights)

    def prepare_input_first_layer(self, x):
        input_layer = []
        for n in range(self.conv_depth[0]):
            input_layer.append(x)
        self.layer_inputs[0] = input_layer

    def initialize_network_weights(self):
        # Calculation of weight size for convolution layer
        size_FC = 161
        for k in range(len(self.layer_type)):
            if self.layer_type[k] == 1:
                if size_FC % self.pooling_size != 0:
                    size_FC = size_FC / self.pooling_size + 1
                else:
                    size_FC /= self.pooling_size
            if self.layer_type[k] == 2:
                self.filter_size[k] = (size_FC, 28)

        for i in range(len(self.layer_type)):
            if self.layer_type[i] == 1:
                self.weights[i] = 0
            else:
                self.weights[i] = initialize_conv_weights.init_weights(self.filter_size[i], self.conv_depth[i])


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
