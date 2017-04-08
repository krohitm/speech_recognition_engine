"""
Script is used to create JSON-description file which is later used to read audio files and corresponding labels.
Written for LibriSpeech dataset format
Input arguments: [path of data directory] [path of output JSON file]
JSON file format:
{"keys": [path of audio file],
 "duration": [audio file duration],
  "text": [corresponding text]}
Courtesy: Baidu Research Github:  https://github.com/baidu-research
"""

import os
import json
import soundfile
import argparse


def main(data_directory, output_file):
    labels = []
    durations = []
    keys = []

    for group in os.listdir(data_directory):
        if group != ".DS_Store":
            speaker_path = os.path.join(data_directory, group)
            for speaker in os.listdir(speaker_path):
                if speaker != ".DS_Store":
                    labels_file = os.path.join(speaker_path, speaker, '{}-{}.trans.txt'.format(group, speaker))
                    for line in open(labels_file):
                        split = line.strip().split()
                        file_id = split[0]
                        label = ' '.join(split[1:]).lower()
                        audio_file = os.path.join(speaker_path, speaker, file_id) + '.flac'
                        keys.append(audio_file)
                        durations.append(float(soundfile.SoundFile(audio_file).__len__()) /
                                         float(soundfile.SoundFile(audio_file).samplerate))
                        labels.append(label)

    with open(output_file, 'w') as out_file:
        for i in range(len(keys)):
            if durations[i] < 1.50:
                line = json.dumps({'keys': keys[i], 'duration': durations[i], 'text': labels[i]})
                out_file.write(line + '\n')


if __name__ == '__main__':
    # path_data_directory = "../MLProject/LibriSpeech/dev-clean/"
    # path_output_file = "./data.json"
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    parser.add_argument('output_file', type=str,
                        help='Path to output file')
    args = parser.parse_args()
    main(args.data_directory, args.output_file)
