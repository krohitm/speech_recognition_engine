"""
Class sused to extract features from audio file which is later used by the network for training and testing
Reference: Github: baidu-reaserch/ba-dls-deepspeech
"""
import json
import numpy as np
import logging.config
from utils import spectrogram_from_file
from utils import text_to_int_sequence
from utils import normalize_features

logger = logging.getLogger(__name__)


class DataGenerator(object):
    def __init__(self, step=10, window=20, desc_file=None):
        """
        :param step: step size in milliseconds between windows
        :param window: FFT window size in millisecond
        :param desc_file: Path of JSON file containing labels and paths to the audio files
        """
        self.step = step
        self.window = window
        if desc_file is not None:
            self.load_metadata_from_descfile(desc_file)

    def generate_audio_features(self, audio_clip):
        return spectrogram_from_file(audio_clip, self.step, self.window)

    def load_metadata_from_descfile(self, desc_file, partition='train'):
        """
        :param desc_file: Path to JSON file containing labels and path to audio files
        :param partition: One of 'dev','train','validation' and 'test'
        :return:
        """
        logger.info('Reading description file: {} for partition {}'
                    .format(desc_file, partition))
        audio_paths, durations, texts = [], [], []
        with open(desc_file) as json_file:
            for line_num, json_line in enumerate(json_file):
                try:
                    spec = json.loads(json_line)
                    audio_paths.append(spec['keys'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text'])
                except Exception as e:
                    logger.warn('Error reading in line #{}: {}'
                                .format(line_num, json_line))

        if partition == 'dev':
            self.dev_audio_path = audio_paths
            self.dev_durations = durations
            self.dev_texts = texts
        else:
            raise Exception("Invalid partition to load metadata."
                            "Must be dev/train/validation/test")

    def load_dev_data(self, desc_file):
        self.load_metadata_from_descfile(desc_file, 'dev')

    def generate_minibatch(self, audio_paths, texts):
        """
        Function generates mini-btaches for further processing
        :param self:
        :param audio_paths: Paths of audio files (list format)
        :param texts: Texts corresponding to audio files (list format)
        :return:
        """
        assert len(audio_paths) == len(texts), "Input and output length should match"
        features = [self.generate_audio_features(a) for a in audio_paths]
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        x = np.zeros((mb_size, max_length, feature_dim))
        y = []
        label_lengths = []
        for i in range(mb_size):
            feat = features[i]
            feat = normalize_features(feat)
            x[i, :feat.shape[0], :] = feat
            label = text_to_int_sequence(texts[i])
            y.append(label)
            label_lengths.append(len(label))
        # Flatten labels to comply with CTC signature
        y = reduce(lambda x, z: x + z, y)
        return {
            'x': x,  # features(padded with zero) shape(minibatch-size, time-steps, feature-dimensions)
            'y': y,  # Flattened output labels transformed as integers
            'texts': texts,  # original list of texts
            'input-lengths': input_lengths,  # length of each input
            'label-lengths': label_lengths  # length of each label
        }

    def iterate(self, audio_paths, texts, batch_size):
        """

        :param audio_paths:
        :param texts:
        :param batch_size:
        :return:
        """
        start = 0
        print batch_size
        number_of_iterations = int(np.ceil(len(audio_paths)/batch_size))
        for k in range(number_of_iterations):
            print audio_paths[start: start+batch_size]
            print texts[start: start+batch_size]
            yield self.generate_minibatch(audio_paths[start: start+batch_size],
                                          texts[start: start+batch_size])

    def iterate_dev(self, mini_batch_size=16, sortBy_duration=False):
        """
        Function iterates over the training data and form mini-batches for further processing
        :param mini_batch_size: batch size
        :param sortBy_duration: if true, sort audio signals by duration before forming mini batches
        :return:
        """
        durations, audio_paths, texts = (self.dev_durations,
                                         self.dev_audio_path,
                                         self.dev_texts)
        if sortBy_duration:
            durations, audio_paths, texts = zip(*sorted(zip(durations, audio_paths, texts)))
            audio_paths = list(audio_paths)
            texts = list(texts)
        return self.iterate(audio_paths, texts, mini_batch_size)
