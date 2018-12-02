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

        for epoch in range(0, 10):
            logging.info('Starting {} Epoch Training!'
                         .format(epoch))
            np.random.shuffle(train)

            for i in range(0, len(train), self.C_PARAMS.baths):
                indexes = train[list(range(
                    i, i + self.C_PARAMS.baths))]
                self.__step_process(i, indexes)

            self.__evaluate(validation)

    # Train, it's 2-D arrays already separated
    # to the timeline. Validation 1-D array.
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

    # Receive working Indexed and Step, later mapping Values
    # into Tuple x_y_s = x input, y output, s current step
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
            self.model = Sequential()
            self.model.add(InputLayer(input_shape=(
                len(self.P_PARAMS.backward), x_y[0].shape[2])))

            self.model.add(
                LSTM(units=128, return_sequences=True,
                     kernel_initializer=he_normal(),
                     activation=tanh))

            self.model.add(
                LSTM(units=64, return_sequences=True,
                     kernel_initializer=he_normal(),
                     activation=tanh))

            self.model.add(
                LSTM(units=32, return_sequences=False,
                     kernel_initializer=he_normal(),
                     activation=tanh))

            self.model \
                .add(Dense(units=1,
                           kernel_initializer=he_normal(),
                           activation=linear))
            self.model \
                .compile(loss=mean_squared_error,
                         optimizer=Adam(lr=0.1))

        self.model.train_on_batch(
            x_y[0], x_y[1])

    def __evaluate(self, validation):

        def local_save(x_y):
            evaluation = self.model\
                .evaluate(x_y[0], x_y[1])
            logging.info(
                "Final validation Evaluation On Items Size: {} Value: {}"
                    .format(len(x_y[0]), evaluation))
            self.VISUAL.set_evaluation(evaluation)

        Preprocessor(self.P_PARAMS).build(
            '../' + Settings.TRAIN_FRAMES,
            '../' + Settings.TRAIN_Y, validation) \
            .subscribe(local_save)


#####################################
if __name__ == "__main__":

    def combine_workers():
        return []

    workers = combine_workers()
    workers.append(MiniBatchWorker(
        PreprocessorParams(backward=(0, 1, 2), frame_y_trim=(
            100, 350), frame_x_trim=(230, 360), frame_scale=1),
        ControllerParams(baths=10, train_part=0.9, step_vis=150, samples=20400)))

    for worker in workers:
        worker.run()

        def worker_plot():
            fig, ax = plt.subplots()
            ax.plot(worker.VISUAL.iters,
                    worker.VISUAL.costs)

            ax.set(xlabel='Num of Iter (I)',
                   ylabel='Costs (J)')
            ax.grid()
            return plt

        plot = worker_plot()

        Helper.save_plot_with(
            '../' + Settings.BUILD_PATH, plot,
            'V1', worker.model, worker.P_PARAMS)
