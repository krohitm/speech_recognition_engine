"""
Common functions are written here
"""

import logging.config
import soundfile as sf
import numpy as np
from scipy.signal import spectrogram
from sklearn.preprocessing import normalize

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Test info statement")
logger.debug("Test debug statement")
logger.warn("Test warn statement")
logger.error("Test error statement")


def spectrogram_from_file(fileName, step, window):
    """
    Function used to compute FFT from raw audio signal
    :param fileName: path of individual files
    :param step: step size in milliseconds between windows
    :param window: window-size in milliseconds of FFT
    :return:
    """
    logger.debug("Generating spectrogram from audio file.")
    with sf.SoundFile(fileName) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)
        f, t, x = spectrogram(audio, fs=sample_rate, window=('tukey', 0.0), nperseg=fft_length, noverlap=hop_length)
        return np.transpose(x)


def text_to_int_sequence(text):
    """
    Converts text to integer sequence using a character map
    :param text:
    :return:
    """
    int_sequence = []
    for c in text:
        if c == ' ':
            int_sequence.append(2)
        elif c == "'":
            int_sequence.append(1)
        else:
            # code changed to start alphabets from 3 rather than 1
            int_sequence.append(ord(c) - 94)
    return int_sequence


def normalize_features(feat):
    return normalize(feat, axis=0)
