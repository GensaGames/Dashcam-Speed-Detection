import os
import sys
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
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, TimeDistributed
from keras.layers import SimpleRNN
from keras.losses import mean_squared_error
from keras.activations import tanh, linear, sigmoid, relu
from keras.optimizers import RMSprop, SGD, Adadelta, Adam
from keras.initializers import RandomUniform, he_normal
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import InputLayer

from app.core.Parameters import ControllerParams, \
    VisualHolder, PreprocessorParams
from app.core.Preprocessing import Preprocessor


class MiniBatchWorker:

    def __init__(self, p_params, c_params):
        self.P_PARAMS, self.C_PARAMS, self.VISUAL \
            = p_params, c_params, VisualHolder()
        self.model = None

    def run(self):
        train, validation = \
            self.split_indexes()

        for e in range(6):
            np.random.shuffle(train)
            logging.info('Starting {} Training Epoch!'
                         .format(str(e)))

            for i in range(0, len(train), self.C_PARAMS.baths):
                indexes = train[list(range(
                    i, i + self.C_PARAMS.baths))]
                self.__step_process(i, indexes)

            self.__evaluate(validation)

    def split_indexes(self):
        indexes = np.arange(
            max(self.P_PARAMS.backward), self.C_PARAMS.samples)

        assert 0 < self \
            .C_PARAMS.train_part < 1
        max_train_index = int(
            self.C_PARAMS.train_part * len(indexes))

        max_train_index = self.C_PARAMS.baths * int(
            max_train_index / self.C_PARAMS.baths)

        train = indexes[:max_train_index]
        return train,\
               indexes[max_train_index:]

    def __step_process(self, step, indexes):
        obs = Preprocessor(self.P_PARAMS).build(
            '../' + Settings.TRAIN_FRAMES,
            '../' + Settings.TRAIN_Y, indexes) \
            .publish()

        obs\
            .subscribe(self.__step_model)

        obs\
            .filter(lambda _: step > self.C_PARAMS.step_vis and (
                    step % self.C_PARAMS.step_vis == 0 or
                    step >= self.C_PARAMS.samples - self.C_PARAMS.baths)) \
            .map(lambda x_y: (x_y[0], x_y[1], step)) \
            .subscribe(self.__step_visual)

        obs.connect()

    def __step_visual(self, x_y_s):
        cost = self.model \
            .evaluate(x_y_s[0], x_y_s[1])

        logging.info("Added for Visualisation. Iter: {} Cost: {}"
              .format(x_y_s[2], cost))
        self.VISUAL.add(x_y_s[2], cost)

    def __step_model(self, x_y):
        if self.model is None:
            input_shape = (x_y[0].shape[2],
                x_y[0].shape[3], 1)

            convolution = Sequential()
            convolution.add(Conv2D(
                filters=36, kernel_size=(5, 5), activation=sigmoid,
                padding='valid', input_shape=input_shape,
                data_format='channels_last'))

            convolution.add(Conv2D(
                filters=12, kernel_size=(3, 3), activation=sigmoid,
                padding='valid', input_shape=input_shape,
                data_format='channels_last'))

            convolution.add(MaxPooling2D(pool_size=(2, 2)))
            convolution.add(Flatten())

            self.model = Sequential()
            self.model.add(TimeDistributed(convolution))

            self.model.add(
                LSTM(units=36, return_sequences=True,
                     kernel_initializer=he_normal()))

            self.model.add(
                LSTM(units=12, return_sequences=False,
                          kernel_initializer=he_normal()))

            self.model \
                .add(Dense(units=1,
                           kernel_initializer=he_normal(),
                           activation=linear))
            self.model \
                .compile(loss=mean_squared_error,
                         optimizer=Adam(lr=1))

        logging.info(
            'Training Batch loss: {}'.format(
                self.model.train_on_batch(x_y[0], x_y[1])))

    def __evaluate(self, validation):

        def local_save(x_y):
            logging.info("Starting Cross Validation...")

            evaluation = self.model\
                .evaluate(x_y[0], x_y[1])

            logging.info("Cross Validation Done on "
                         "Items Size: {} Value: {}".format(
                len(x_y[0]), evaluation))
            self.VISUAL.set_evaluation(evaluation)

        Preprocessor(self.P_PARAMS).build(
            '../' + Settings.TRAIN_FRAMES,
            '../' + Settings.TRAIN_Y, validation[:10]) \
            .subscribe(local_save)


#####################################
if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    def combine_workers():
        workers = [MiniBatchWorker(
            PreprocessorParams(
                backward=(0, 1, 2), frame_y_trim=(190, -190),
                frame_x_trim=(220, -220), frame_scale=1.5),
            ControllerParams(
                baths=10, train_part=0.9, step_vis=150,
                samples=20400))]
        return workers


    def worker_plot(worker):
        fig, ax = plt.subplots()
        ax.plot(worker.VISUAL.iters,
                worker.VISUAL.costs)

        ax.set(xlabel='Num of Iter (I)',
               ylabel='Costs (J)')
        ax.grid()
        return plt

    def start_train():
        for worker in combine_workers():
            worker.run()

            Helper.save_plot_with(
                '../' + Settings.BUILD_PATH, worker_plot(worker),
                'V1', worker.model, worker.P_PARAMS)
            plt.show()

    start_train()

