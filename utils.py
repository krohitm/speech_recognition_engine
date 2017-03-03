"""
Common functions are written here
"""

from logger import log
import soundfile as sf
import numpy as np
from scipy.signal import spectrogram
from sklearn.preprocessing import normalize


def spectrogram_from_file(fileName, step, window):
    """
    Function used to compute FFT from raw audio signal
    :param fileName: path of individual files
    :param step: step size in milliseconds between windows
    :param window: window-size in milliseconds of FFT
    :return:
    """
    log.debug("Inside spectrogram_from_file")
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
    log.debug("Inside text_to_int_sequence")
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


def weights_random(layer1_size, layer2_size):
    e = np.sqrt(6) / np.sqrt(layer1_size + layer2_size)
    return np.dot(np.random.rand(layer1_size, layer2_size), 2 * e) - e