import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.activations import linear, sigmoid, relu
from keras.initializers import he_normal
from keras.layers import Dense, Flatten, Dropout, Conv3D, MaxPooling3D, Lambda, Convolution2D, ELU
from keras.losses import mean_squared_error
from keras.optimizers import Adam

import app.Settings as Settings
import app.other.Helper as Helper
from app.core import Augmenters
from app.core.Parameters import ControllerParams, \
    VisualHolder, PreprocessorParams
from app.core.Preprocessing import Preprocessor


class MiniBatchWorker:

    def __init__(self, p_params, c_params, model=None):
        self.P_PARAMS, self.C_PARAMS, self.VISUAL \
            = p_params, c_params, VisualHolder()
        self.model = model

    def start_epochs(self):
        train, validation = \
            self.__split_indexes()

        for e in range(self.C_PARAMS.epochs):
            logger.info('Starting {} Training Epoch!'
                        .format(str(e)))
            self.__start_train(train, validation)

    def show_evaluation(self):
        train, validation = \
            self.__split_indexes()

        def local_evaluate(x_y):
            cost = self.model \
                .evaluate(x_y[0], x_y[1])

            logger.info("Evaluation on Items: {} Cost: {}"
                        .format(len(x_y[0]), cost))

        while True:
            np.random.shuffle(validation)
            Preprocessor(self.P_PARAMS, Augmenters.get_new_validation()).build(
                '../' + Settings.TRAIN_FRAMES,
                '../' + Settings.TRAIN_Y, validation[:100]) \
                .subscribe(local_evaluate)

    def make_test(self):
        samples = np.arange(
            max(self.P_PARAMS.backward), 10798)

        Helper.clear_built_test(
            '../' + Settings.BUILD,
            self.P_PARAMS.backward)

        def local_evaluate(x_y):
            predictions = self.model \
                .predict(x_y[0])

            Helper.add_built_test(
                '../' + Settings.BUILD,
                predictions)

        BATCHES = 120

        for i in range(0, len(samples), BATCHES):
            logger.info('Moving to next Step-Idx {}.'
                        .format(str(i)))
            step = i + BATCHES if i + BATCHES < len(
                samples) else len(samples)

            samples_step = samples[list(range(i, step))]
            Preprocessor(self.P_PARAMS, Augmenters.get_new_validation())\
                .build('../' + Settings.TEST_FRAMES, None,
                       samples_step)\
                .subscribe(local_evaluate)

    def restore_backup(self):
        if self.model is not None:
            logging.error(
                'Model already created. Do not override!')
            return

        try:
            self.model, self.P_PARAMS, self.C_PARAMS, self.VISUAL = \
                Helper.restore_model_with(
                    '../' + Settings.BUILD, self.C_PARAMS.name)
        except FileNotFoundError:
            logging.error(
                'Do not have Backup! Starting new.')

    def __start_train(self, train, validation):
        np.random.shuffle(train)

        for i in range(0, len(train), self.C_PARAMS.baths):
            indexes = train[list(range(
                i, i + self.C_PARAMS.baths))]
            self.__step_process(i, indexes)

        self.__evaluate(validation)
        self.do_backup()

    def do_backup(self):
        Helper.backup_model_with(
            '../' + Settings.BUILD, self.C_PARAMS.name,
            self.model, self.P_PARAMS, self.C_PARAMS, self.VISUAL)
        pass

    def __split_indexes(self):
        indexes = np.arange(
            max(self.P_PARAMS.backward), self.C_PARAMS.samples)

        assert 0 < self \
            .C_PARAMS.train_part < 1
        max_train_index = int(
            self.C_PARAMS.train_part * len(indexes))

        max_train_index = self.C_PARAMS.baths * int(
            max_train_index / self.C_PARAMS.baths)

        train = indexes[:max_train_index]
        return train, indexes[max_train_index:]

    def __step_process(self, step, indexes):
        obs = Preprocessor(self.P_PARAMS,
                           Augmenters.get_new_training()).build(
            '../' + Settings.TRAIN_FRAMES,
            '../' + Settings.TRAIN_Y, indexes) \
            .publish()

        obs.filter(lambda _: step > self.C_PARAMS.step_vis and (
                step % self.C_PARAMS.step_vis == 0 or
                step >= self.C_PARAMS.samples - self.C_PARAMS.baths)) \
            .map(lambda x_y: (x_y[0], x_y[1], step)) \
            .subscribe(self.__step_visual)

        obs.subscribe(self.__step_model)
        obs.connect()

    def __step_visual(self, x_y_s):
        cost = self.model \
            .evaluate(x_y_s[0], x_y_s[1])

        logger.info("Added for Visualisation. Iter: {} Cost: {}"
                    .format(x_y_s[2], cost))
        self.VISUAL.add_iter(x_y_s[2], cost)

    def __step_model(self, x_y):
        if self.model is None:

            input_shape = (
                len(self.P_PARAMS.backward) - 1,
                x_y[0].shape[2], x_y[0].shape[3], x_y[0].shape[4])

            self.model = Sequential()
            self.model.add(
                Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2),
                       input_shape=input_shape, padding='same',
                       kernel_initializer=he_normal(),
                       data_format='channels_last'))

            self.model.add(ELU())
            self.model.add(
                Conv3D(filters=48, kernel_size=(3, 5, 5), strides=(1, 2, 2),
                       input_shape=input_shape, padding='same',
                       kernel_initializer=he_normal(),
                       data_format='channels_last'))

            self.model.add(ELU())
            self.model.add(
                Conv3D(filters=48, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                       input_shape=input_shape, padding='valid',
                       kernel_initializer=he_normal(),
                       data_format='channels_last'))

            self.model.add(ELU())
            self.model.add(
                Conv3D(filters=64, kernel_size=(1, 3, 3), strides=(1, 1, 1),
                       input_shape=input_shape, padding='valid',
                       kernel_initializer=he_normal(),
                       data_format='channels_last'))

            self.model.add(ELU())
            self.model.add(MaxPooling3D(pool_size=(1, 2, 2)))
            self.model.add(Flatten())

            self.model \
                .add(Dense(units=256,
                           kernel_initializer=he_normal()))
            self.model.add(ELU())

            self.model \
                .add(Dense(units=128,
                           kernel_initializer=he_normal()))
            self.model.add(ELU())

            self.model \
                .add(Dense(units=64,
                           kernel_initializer=he_normal()))
            self.model.add(ELU())

            self.model \
                .add(Dense(units=1,
                           kernel_initializer=he_normal(),
                           activation=linear))
            self.model \
                .compile(loss=mean_squared_error,
                         optimizer=Adam(lr=1e-4))

            # Comment/Uncomment for showing detailed
            # info about Model Structure.

            # from keras.utils import plot_model
            # plot_model(self.model, to_file='model_plot1.png',
            #            show_shapes=True, show_layer_names=True)

        value = self.model.train_on_batch(x_y[0], x_y[1])
        logger.debug('Training Batch loss: {}'
                     .format(value))

    def __evaluate(self, validation):
        np.random.shuffle(validation)

        def local_save(x_y):
            logger.info("Starting Cross Validation.")

            evaluation = self.model \
                .evaluate(x_y[0], x_y[1])

            logger.info(
                "Cross Validation Done on Items Size: {} "
                "Value: {}".format(len(x_y[0]), evaluation))
            self.VISUAL.add_evaluation(evaluation)

        Preprocessor(self.P_PARAMS,
                     Augmenters.get_new_training()).build(
            '../' + Settings.TRAIN_FRAMES,
            '../' + Settings.TRAIN_Y, validation[:100]) \
            .subscribe(local_save)


#####################################
if __name__ == "__main__":

    def set_logger():
        sys.setrecursionlimit(1001001)
        formatter = logging.Formatter(
            '%(asctime)-15s %(message)s')

        log = logging.getLogger(
            os.path.basename(__file__))
        log.setLevel(logging.DEBUG)

        path_to = '../' + Settings.BUILD
        if not os.path.exists(path_to):
            os.makedirs(path_to)

        # Comment/Uncomment in case of
        # using loggin in files
        # handler = logging.FileHandler(
        #     filename=path_to + Settings.NAME_LOGS)
        # handler.setLevel(logging.DEBUG)
        # handler.setFormatter(formatter)
        # log.addHandler(handler)

        # Comment/Uncomment in case of
        # issue with logging to system

        handler1 = logging.StreamHandler()
        handler1.setLevel(logging.DEBUG)
        handler1.setFormatter(formatter)
        log.addHandler(handler1)

        return log

    logger = set_logger()

    def combine_workers():
        workers = [MiniBatchWorker(
            PreprocessorParams(
                backward=(0, 1, 2, 3), frame_y_trim=(160, -160),
                frame_x_trim=(110, -110), frame_scale=0.8,
                area_float=6),
            ControllerParams(
                'OPT-V7-OPT-3D-CNN/', baths=20, train_part=0.65,
                epochs=15, step_vis=200, samples=20400))]
        return workers

    def worker_plot(worker):
        fig, ax = plt.subplots()

        ax.plot(
            range(0, len(worker.VISUAL.evaluations)),
            worker.VISUAL.evaluations)

        ax.set(xlabel='Iters (I)',
               ylabel='Costs (J)')
        ax.grid()

        plt.savefig(
            '../' + Settings.BUILD + '/' + Settings.MODELS + '/'
            + worker.C_PARAMS.name + '/' + Settings.NAME_MODEL_PLOT)
        return plt

    def start_train():
        for worker in combine_workers():
            worker.restore_backup()
            worker.start_epochs()
            # worker.show_evaluation()
            # worker.make_test()
            # worker_plot(worker)


    start_train()
