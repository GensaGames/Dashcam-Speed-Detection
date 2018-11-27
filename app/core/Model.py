import numpy as np

import app.core.Parameters
import app.Settings as Settings
import app.other.Helper as Helper
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools
import logging
from keras import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.losses import mean_squared_error
from keras.activations import tanh, linear, sigmoid, relu
from keras.optimizers import RMSprop, SGD, Adadelta
from keras.initializers import RandomUniform
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import InputLayer

from app.core.Parameters import ControllerParams, \
    VisualHolder, PreprocessorParams
from app.core.Preprocessing import Preprocessor


model = Sequential()
model.add(InputLayer(input_shape=(1, 20)))

print('ddd')
model.add(
    SimpleRNN(units=64, return_sequences=True,
              kernel_initializer=RandomUniform(-0.05, 0.05),
              activation=sigmoid))

print('ddd 22')
model.add(
    SimpleRNN(units=64,
              kernel_initializer=RandomUniform(-0.05, 0.05),
              activation=sigmoid))

model \
    .add(Dense(units=1,
               kernel_initializer=RandomUniform(-0.05, 0.05),
               activation=linear))
model \
    .compile(loss=mean_squared_error,
             optimizer=SGD(lr=0.001))


