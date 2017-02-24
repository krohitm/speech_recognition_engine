"""
Class sused to extract features from audio file which is later used by the network for training and testing
Reference: Github: baidu-reaserch/ba-dls-deepspeech
"""
import json
import logging.config
from utils import spectrogram_from_file

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
        spectrogram_from_file(audio_clip, self.step, self.window)

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