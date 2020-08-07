import os

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model

import app.other.Helper as Helper
from app import Settings
from app.core import Augmenters
from app.core.Models import Models
from app.core.Parameters import ControllerParams, \
    VisualHolder, PreprocessorParams
from app.core.Preprocessing import Preprocessor
from app.other.LoggerFactory import get_logger


class MiniBatchWorker:

    def __init__(self, p_params, c_params):
        self.P_PARAMS, self.C_PARAMS, self.VISUAL \
            = p_params, c_params, VisualHolder()
        self.model = None

    def start_training_epochs(self):
        train, validation = \
            self.__prepare_training_data()

        for e in range(self.C_PARAMS.epochs):
            get_logger().info('Starting {} Training Epoch!'
                              .format(str(e)))

            np.random.shuffle(train)
            self.__start_train(train, validation)
            get_logger().info("Epoch {} training done. Backup."
                              .format(e))

        get_logger().info('Starting Final Validation!')
        for i in range(10):
            self.__step_validation(validation)

    # Take more important data from the resources,
    # where car was on the street roads, and has more
    # variance
    def __prepare_training_data(self):
        indexes = np.arange(
            max(self.P_PARAMS.backward), self.C_PARAMS.samples)

        assert 0 < self \
            .C_PARAMS.train_part < 1
        max_initial_idx = int(
            self.C_PARAMS.train_part * len(indexes))

        train = np.concatenate((
            indexes[:max_initial_idx], self.__get_new_stop_frames()))

        # Just align with exact part of batches.
        max_train_idx = self.C_PARAMS.baths * int(
            len(train) / self.C_PARAMS.baths)
        train = train[:max_train_idx]

        return train, indexes[max_initial_idx:]

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

    def __start_train(self, train, validation):
        for step in range(0, len(train) // self.C_PARAMS.baths):
            get_logger().info("Start Train step: {}."
                              .format(step))

            indexes = train[list(range(
                step * self.C_PARAMS.baths,
                (step + 1) * self.C_PARAMS.baths
            ))]

            self.__step_process(
                step, indexes, validation)

    def __step_process(self, step, indexes, validation):

        Preprocessor(
            self.P_PARAMS,
            Augmenters.get_new_training()
        ).build(
            indexes,
            (Settings.TRAIN_FRAMES, Settings.TRAIN_Y)
        ).subscribe(self.__step_model, on_error=lambda e: get_logger()
                    .error('Exception! ' + str(e)))

        if step > 0 and (
                step % self.C_PARAMS.step_vis == 0 or
                step >= self.C_PARAMS.samples - self.C_PARAMS.baths):
            self.__step_validation(validation)
            self.do_backup()

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

            Preprocessor(
                self.P_PARAMS,
                Augmenters.get_new_validation()
            ).build(
                samples_step,
                (Settings.TEST_FRAMES, None)
            ).subscribe(local_evaluate, on_error=lambda e: get_logger()
                        .error('Exception! ' + str(e)))

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
            self.model = Models.get3D_CNN(x_y[0])
            plot_structure(self)

        value = self.model.train_on_batch(x_y[0], x_y[1])
        self.VISUAL.add_training_point(value)

        get_logger().debug('Training Batch loss: {}'
                           .format(value))

    def __step_validation(self, validation):
        get_logger().info("Starting Cross Validation.")
        np.random.shuffle(validation)

        def local_save(x_y):
            mse = self.model \
                .evaluate(x_y[0], x_y[1])

            get_logger().info(
                "Cross Validation Done on Items Size: {} "
                "MSE: {}".format(len(x_y[0]), mse))

            self.VISUAL.add_validation_point(mse)

        Preprocessor(
            self.P_PARAMS,
            Augmenters.get_new_validation()
        ).build(
            validation[:200],
            (Settings.TRAIN_FRAMES, Settings.TRAIN_Y)
        ).subscribe(local_save, on_error=lambda e: get_logger()
                    .error('Exception! ' + str(e)))


#####################################
if __name__ == "__main__":

    def combine_workers():
        workers = [MiniBatchWorker(
            PreprocessorParams(
                backward=(0, 1, 2, 3),
                frame_y_trim=(230, -160),
                frame_x_trim=(180, -180),
                frame_scale=1,
            ),
            ControllerParams(
                'NEW-OPT-A71',
                baths=30,
                train_part=0.7,
                epochs=1,
                step_vis=10,
                samples=20400))]
        return workers


    def plot_structure(worker):
        plot_model(
            worker.model,
            to_file=Settings.NAME_STRUCTURE,
            show_shapes=True,
            show_layer_names=True
        )
        return plt


    def plot_progress(worker):
        fig, ax = plt.subplots()

        ax.plot(
            list(map(
                lambda x: (x + 1) * worker.C_PARAMS.step_vis,
                list(range(0, len(worker.VISUAL.points_validation)))
            )), worker.VISUAL.points_validation)

        ax.plot(
            list(
                range(0, len(worker.VISUAL.points_training))
            ),
            worker.VISUAL.points_training)

        ax.legend(['Validation', 'Training'])
        ax.set_ylim([0, 40])
        ax.set(
            xlabel='Batch Step (S)',
            ylabel='Errors (J)'
        )
        ax.grid()

        plt.savefig('/'.join([
            Settings.BUILD,
            Settings.MODELS,
            worker.C_PARAMS.name,
            Settings.NAME_MODEL_PLOT
        ]))
        return plt


    def start_actions():
        for worker in combine_workers():
            worker.restore_backup()
            """ 
            Comment/Uncomment for making different
            controller actions.
            """
            worker.start_training_epochs()
            # worker.start_evaluation()
            # worker.create_test_output()
            plot_progress(worker)


    start_actions()
