import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.activations import linear, sigmoid, relu
from keras.initializers import he_normal
from keras.layers import Dense, Flatten, Dropout, Conv3D, MaxPooling3D, Lambda, Convolution2D, ELU, BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam

import app.Settings as Settings
import app.other.Helper as Helper
from app.core import Augmenters
from app.core.Parameters import ControllerParams, \
    VisualHolder, PreprocessorParams
from app.core.Preprocessing import Preprocessor
from app.other.LoggerFactory import get_logger


class MiniBatchWorker:

    def __init__(self, p_params, c_params, model=None):
        self.P_PARAMS, self.C_PARAMS, self.VISUAL \
            = p_params, c_params, VisualHolder()
        self.model = model

    def start_training_epochs(self):
        train, validation = \
            self.__prepare_training_data()

        for e in range(self.C_PARAMS.epochs):
            logger.info('Starting {} Training Epoch!'
                        .format(str(e)))

            np.random.shuffle(train)
            self.__start_train(train, validation)

    PREFIX_STOP_SIZE = int(10.e+4)

    # Include special located frames with car
    # stop and this variance variance
    def __get_new_stop_frames(self):
        source_stop_frames = Settings.TRAIN_FRAMES_STOP
        stop_indexes = []

        for next_dir in os.listdir(source_stop_frames):
            for idx, _ in enumerate(os.listdir(
                    source_stop_frames + '/' + next_dir)):

                if idx < max(self.P_PARAMS.backward):
                    continue

                # Just use folder as prefix for index, and avoid
                # collision with initial train indexes from train part
                idx_ = int(next_dir) * self.PREFIX_STOP_SIZE + idx
                stop_indexes.append(idx_)

        np.random.shuffle(stop_indexes)
        return stop_indexes

    # Take more important data from the resources,
    # where car was on the street roads, and has more
    # variance
    def __prepare_training_data(self):
        indexes = np.arange(
            max(self.P_PARAMS.backward), self.C_PARAMS.samples)
        np.random.shuffle(indexes)

        assert 0 < self \
            .C_PARAMS.train_part < 1
        max_initial_idx = int(
            self.C_PARAMS.train_part * len(indexes))

        train = np.concatenate((
            indexes[:max_initial_idx], self.__get_new_stop_frames()))
        np.random.shuffle(train)

        # Just align with exact part of batches.
        max_train_idx = self.C_PARAMS.baths * int(
            len(train) / self.C_PARAMS.baths)
        train = train[:max_train_idx]

        return train, indexes[max_initial_idx:]

    def __start_train(self, train, validation):
        step = 0
        for i in range(0, len(train), self.C_PARAMS.baths):
            logger.info("Start Train step: {}.".format(step))

            indexes = train[list(range(
                i, i + self.C_PARAMS.baths))]

            self.__step_process(
                step, indexes, validation)
            step += 1

        logger.info("Epoch training done. Backup.")

    def __step_process(self, step, indexes, validation):
        obs = Preprocessor(self.P_PARAMS, Augmenters
                           .get_new_training()) \
            .set_source(Settings.TRAIN_FRAMES, Settings.TRAIN_Y) \
            .build(indexes) \
            .publish()

        obs.subscribe(self.__step_model)
        obs.connect()

        if step > 0 and (
                step % self.C_PARAMS.step_vis == 0 or
                step >= self.C_PARAMS.samples - self.C_PARAMS.baths):
            self.__evaluate(validation)
            self.do_backup()

    def start_evaluation(self):
        train, validation = \
            self.__prepare_training_data()
        np.random.shuffle(train)

        def local_evaluate(x_y):
            cost = self.model \
                .evaluate(x_y[0], x_y[1])

            logger.info("Evaluation on Items: {} Cost: {}"
                        .format(len(x_y[0]), cost))

        while True:
            np.random.shuffle(validation)
            Preprocessor(self.P_PARAMS, Augmenters
                         .get_new_validation()) \
                .set_source(Settings.TRAIN_FRAMES, Settings.TRAIN_Y) \
                .build(validation[:100]) \
                .subscribe(local_evaluate)

    BATCHES = 120

    def create_test_output(self):
        logger.info("Create model test values.")
        samples = np.arange(
            max(self.P_PARAMS.backward), 10798)

        Helper.clear_built_test(
            Settings.BUILD, self.C_PARAMS.name,
            self.P_PARAMS.backward)

        def local_evaluate(x_y):
            predictions = self.model \
                .predict(x_y[0])

            Helper.add_built_test(
                Settings.BUILD, self.C_PARAMS.name,
                predictions)

        for i in range(0, len(samples), self.BATCHES):
            logger.info('Moving to next Step-Idx {}.'
                        .format(str(i)))
            step = i + self.BATCHES if i + self.BATCHES < len(
                samples) else len(samples)

            samples_step = samples[list(range(i, step))]
            Preprocessor(self.P_PARAMS, Augmenters
                         .get_new_validation()) \
                .set_source(Settings.TEST_FRAMES, None) \
                .build(samples_step) \
                .subscribe(local_evaluate)

    def restore_backup(self):
        logger.info("Restoring Backup...")
        if self.model is not None:
            logging.error(
                'Model already created. Do not override!')
            return

        try:
            self.model, self.P_PARAMS, self.C_PARAMS, self.VISUAL = \
                Helper.restore_model_with(
                    Settings.BUILD, self.C_PARAMS.name)
        except FileNotFoundError:
            logging.error(
                'Do not have Backup! Starting new.')

    def do_backup(self):
        logger.info("Making Backup...")
        Helper.backup_model_with(
            Settings.BUILD, self.C_PARAMS.name,
            self.model, self.P_PARAMS, self.C_PARAMS, self.VISUAL)

    def __step_model(self, x_y):
        if self.model is None:

            input_shape = (
                x_y[0].shape[1], x_y[0].shape[2],
                x_y[0].shape[3])

            self.model = self.nvidia_model(input_shape)

            # Comment/Uncomment for showing detailed
            # info about Model Structure.

            # from keras.utils import plot_model
            # plot_model(self.model, to_file='model_plot1.png',
            #            show_shapes=True, show_layer_names=True)

        value = self.model.train_on_batch(x_y[0], x_y[1])
        logger.debug('Training Batch loss: {}'
                     .format(value))

    @staticmethod
    def nvidia_model(input_shape):

        model = Sequential()
        # normalization
        # perform custom normalization before lambda layer in network
        model.add(Lambda(lambda x: x / 256, input_shape=input_shape))

        model.add(Convolution2D(48, (5, 5),
                                strides=(2, 2),
                                padding='valid',
                                kernel_initializer='he_normal',
                                name='conv1'))

        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Convolution2D(48, (5, 5),
                                strides=(2, 2),
                                padding='valid',
                                kernel_initializer='he_normal',
                                name='conv3'))
        model.add(ELU())
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Convolution2D(64, (3, 3),
                                strides=(1, 1),
                                padding='valid',
                                kernel_initializer='he_normal',
                                name='conv4'))

        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Convolution2D(64, (3, 3),
                                strides=(1, 1),
                                padding='valid',
                                kernel_initializer='he_normal',
                                name='conv5'))

        model.add(Flatten(name='flatten'))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Dense(256, kernel_initializer='he_normal', name='fc1'))
        model.add(ELU())
        model.add(Dense(128, kernel_initializer='he_normal', name='fc2'))
        model.add(ELU())
        model.add(Dense(64, kernel_initializer='he_normal', name='fc3'))
        model.add(ELU())

        model.add(Dense(1, name='output', kernel_initializer='he_normal'))

        adam = Adam(lr=1e-4)
        model.compile(optimizer=adam, loss='mse')

        return model

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
                     Augmenters.get_new_training()) \
            .set_source(Settings.TRAIN_FRAMES, Settings.TRAIN_Y) \
            .build(validation[:100]) \
            .subscribe(local_save)


#####################################
if __name__ == "__main__":

    logger = get_logger()


    def combine_workers():
        workers = [MiniBatchWorker(
            PreprocessorParams(
                backward=(0, 2), frame_y_trim=(100, -160),
                frame_x_trim=(50, -50), frame_scale=0.6,
                area_float=6),
            ControllerParams(
                'NV-OPT-V600-2D-CNN', baths=30, train_part=0.7,
                epochs=4, step_vis=80, samples=20400))]
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
            Settings.BUILD + '/' + Settings.MODELS + '/'
            + worker.C_PARAMS.name + '/' + Settings.NAME_MODEL_PLOT)
        return plt


    def start_train():
        for worker in combine_workers():
            worker.restore_backup()
            worker.start_training_epochs()
            # worker.start_evaluation()
            # worker.create_test_output()
            # worker_plot(worker)


    start_train()
