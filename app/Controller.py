import os
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.activations import linear
from keras.initializers import he_normal
from keras.layers import Dense, Flatten, Dropout, Conv2D, Conv3D, ELU, BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam

from app import Settings
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
            get_logger().info('Starting {} Training Epoch!'
                        .format(str(e)))

            np.random.shuffle(train)
            self.__start_train(train, validation)

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
                idx_ = int(next_dir) * Settings.PREFIX_STOP_SIZE + idx
                stop_indexes.append(idx_)

        stop_indexes = np.array(stop_indexes, dtype=np.int32)
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
            get_logger().info("Start Train step: {}.".format(step))

            indexes = train[list(range(
                i, i + self.C_PARAMS.baths))]

            self.__step_process(
                step, indexes, validation)
            step += 1

        get_logger().info("Epoch training done. Backup.")

    def __step_process(self, step, indexes, validation):
        Preprocessor(self.P_PARAMS, Augmenters
                     .get_new_training()) \
            .set_source(Settings.TRAIN_FRAMES, Settings.TRAIN_Y) \
            .build(indexes) \
            .subscribe(self.__step_model, on_error=lambda e:
        get_logger().error('Exception! ' + str(e)))

        if step > 0 and (
                step % self.C_PARAMS.step_vis == 0 or
                step >= self.C_PARAMS.samples - self.C_PARAMS.baths):
            self.__evaluate(validation)
            self.do_backup()

    def start_evaluation(self):
        _, validation = \
            self.__prepare_training_data()

        def local_evaluate(x_y):
            cost = self.model \
                .evaluate(x_y[0], x_y[1])

            get_logger().info("Evaluation on Items: {} Cost: {}"
                        .format(len(x_y[0]), cost))

        while True:
            np.random.shuffle(validation)
            Preprocessor(self.P_PARAMS, Augmenters
                         .get_new_validation()) \
                .set_source(Settings.TRAIN_FRAMES, Settings.TRAIN_Y) \
                .build(validation[:100]) \
                .subscribe(local_evaluate, on_error=lambda e:
            get_logger().error('Exception! ' + str(e)))

    BATCHES = 120

    def create_test_output(self):
        get_logger().info("Create model test values.")
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
            get_logger().info('Moving to next Step-Idx {}.'
                        .format(str(i)))
            step = i + self.BATCHES if i + self.BATCHES < len(
                samples) else len(samples)

            samples_step = samples[list(range(i, step))]
            Preprocessor(self.P_PARAMS, Augmenters
                         .get_new_validation()) \
                .set_source(Settings.TEST_FRAMES, None) \
                .build(samples_step) \
                .subscribe(local_evaluate, on_error=lambda e:
            get_logger().error('Exception! ' + str(e)))

    def restore_backup(self):
        get_logger().info("Restoring Backup...")
        if self.model is not None:
            get_logger().error(
                'Model already created. Do not override!')
            return

        try:
            self.model, self.P_PARAMS, self.C_PARAMS, self.VISUAL = \
                Helper.restore_model_with(
                    Settings.BUILD, self.C_PARAMS.name)
        except FileNotFoundError:
            get_logger().error(
                'Do not have Backup! Starting new.')

    def do_backup(self):
        get_logger().info("Making Backup...")
        Helper.backup_model_with(
            Settings.BUILD, self.C_PARAMS.name,
            self.model, self.P_PARAMS, self.C_PARAMS, self.VISUAL)

    def __step_model(self, x_y):
        if self.model is None:
            input_shape = (
                x_y[0].shape[1],
                x_y[0].shape[2],
                1,)

            self.model = Sequential()
            self.model.add(
                Conv2D(filters=64, kernel_size=(5, 5), strides=(3, 3),
                       input_shape=input_shape, padding='valid',
                       kernel_initializer=he_normal())
            )

            self.model.add(
                Conv2D(filters=86, kernel_size=(5, 5), strides=(3, 3),
                       padding='valid', kernel_initializer=he_normal())
            )

            self.model.add(
                Conv2D(filters=86, kernel_size=(3, 3), strides=(2, 2),
                       padding='valid', kernel_initializer=he_normal())
            )

            self.model.add(
                Conv2D(filters=86, kernel_size=(3, 3), strides=(1, 1),
                       padding='valid', kernel_initializer=he_normal())
            )

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
            """
            Comment/Uncomment for showing detailed
            info about Model Structure.
            """
            # from keras.utils import plot_model
            # plot_model(self.model, to_file='model_plot1.png',
            #            show_shapes=True, show_layer_names=True)

        value = self.model.train_on_batch(x_y[0], x_y[1])
        get_logger().debug('Training Batch loss: {}'
                     .format(value))

    def __evaluate(self, validation):
        np.random.shuffle(validation)

        def local_save(x_y):
            get_logger().info("Starting Cross Validation.")

            evaluation = self.model \
                .evaluate(x_y[0], x_y[1])

            get_logger().info(
                "Cross Validation Done on Items Size: {} "
                "Value: {}".format(len(x_y[0]), evaluation))
            self.VISUAL.add_evaluation(evaluation)

        Preprocessor(self.P_PARAMS,
                     Augmenters.get_new_training()) \
            .set_source(Settings.TRAIN_FRAMES, Settings.TRAIN_Y) \
            .build(validation[:100]) \
            .subscribe(local_save, on_error=lambda e:
        get_logger().error('Exception! ' + str(e)))


#####################################
if __name__ == "__main__":

    def combine_workers():
        workers = [MiniBatchWorker(
            PreprocessorParams(
                backward=(0, 1), frame_y_trim=(100, -170),
                frame_x_trim=(70, -70), frame_scale=1.4,
            ),
            ControllerParams(
                'NEW-OPT-FIN-2', baths=30, train_part=0.99,
                epochs=1, step_vis=40, samples=20400))]
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
            """ 
            Comment/Uncomment for making different
            controller actions.
            """
            worker.restore_backup()
            worker.start_training_epochs()
            # worker.start_evaluation()
            # worker.create_test_output()
            worker_plot(worker)

    start_train()

