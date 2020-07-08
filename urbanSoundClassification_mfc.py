from memory_profiler import memory_usage
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import pandas as pd
from glob import glob
import librosa
import librosa.display
import keras.backend as k
from keras import layers
from keras import models
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
from path import Path
import gc


def create_spectrogram_train(filename, name):
    plt.interactive(False)
    time_series, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=time_series, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = Path(
        '/home/saman/python-projects/data_kaggle/urban_sound_classification/train/train_mfc/'
        + name + '.jpg'
    )
    fig.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, time_series, sample_rate, fig, ax, S


def create_spectrogram_test(filename, name):
    plt.interactive(False)
    time_series, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=time_series, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = Path(
        '/home/saman/python-projects/data_kaggle/urban_sound_classification/test/test_mfc/'
        + name + '.jpg'
    )
    fig.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, time_series, sample_rate, fig, ax, S


data_array = np.array(glob(
    '/home/saman/python-projects/data_kaggle/urban_sound_classification/train/Train/*'
))
i = 0
for path in data_array[i:i+2000]:
    filename, name = path, path.split('/')[-1].split('.')[0]
    create_spectrogram_train(filename, name)
gc.collect()
i = 2000
for path in data_array[i:i+2000]:
    filename, name = path, path.split('/')[-1].split('.')[0]
    create_spectrogram_train(filename, name)
gc.collect()
i = 4000
for path in data_array[i:]:
    filename, name = path, path.split('/')[-1].split('.')[0]
    create_spectrogram_train(filename, name)
gc.collect()
test_array = np.array(glob(
    '/home/saman/python-projects/data_kaggle/urban_sound_classification/test/Test/*'
))
i = 0
for path in test_array[i:i+1500]:
    filename, name = path, path.split('/')[-1].split('.')[0]
    create_spectrogram_test(filename, name)
gc.collect()
i = 1500
for path in test_array[i:]:
    filename, name = path, path.split('/')[-1].split('.')[0]
    create_spectrogram_test(filename, name)
gc.collect()
