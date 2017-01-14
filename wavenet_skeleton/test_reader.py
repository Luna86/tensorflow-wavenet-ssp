import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename

audio_dir = '/home/luna/ssp/data/VCTK-Corpus'
sample_rate = 16000

if __name__ == '__main__':
    iterator = load_generic_audio(audio_dir, sample_rate)
    for audio, filename in iterator:
        print(audio.shape)
        print(audio[1:20, :])
        audio = audio.reshape(-1,1)
        print(audio.shape)