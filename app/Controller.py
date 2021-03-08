import os

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.utils import plot_model

import app.other.Helper as Helper
from app import Settings
from app.Models import Models
from app.Preprocessing import *
from app.other.LoggerFactory import get_logger


class Worker:

    class Params:
        def __init__(self, name, train_part):
            self.name = name
            self.train_part = train_part

    def __init__(self, c_params, p_params):
        self.C_PARAMS = c_params
        self.P_PARAMS = p_params

        self.MODEL = Utils\
            .restore_backup(self.C_PARAMS.name)
        # Checking Backup if already exist
        if not self.MODEL:
            get_logger().warn(
                'Creating New clean Model!'
            )
            # Change New model Structure here
            self.MODEL = Models.nvidia_model()
            Utils.plot_structure(self)
        else:
            get_logger().info(
                'Model was restored from Backup'
            )

    def start_training(self):
        get_logger().info(
            'Start training from Controller!'
        )

        self.MODEL.fit_generator(
            steps_per_epoch=500,
            epochs=85,
            validation_steps=30,
            generator=self.__get_generator(),
            validation_data=self.__get_generator(
                validation=True
            ),
            callbacks=[
                ModelCheckpoint(
                    Helper.get_model_path(
                        self.C_PARAMS.name) + Settings.NAME_MODEL,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1,
                )
            ]
        )

    def __get_generator(self, validation=False):

        data = None
        while True:
            if not data or not data.is_available():
                get_logger().info(
                    'Data Source is not Available! Create new. ' +
                    'Validation? {}'.format(validation)
                )

                data = Data(batches=32).initialize(
                    len(self.P_PARAMS.backward),
                    self.C_PARAMS.train_part
                )

            indexes, source, _ = data.get_train_batch() \
                if not validation else \
                data.get_validation_batch()

            get_logger().debug(
                'Gen Source: {}'.format(source.name)
            )

            x, y = Preprocessor(self.P_PARAMS) \
                .build(indexes, source.path, source.y_values) \
                .run()

            yield x, y


class Utils:

    @staticmethod
    def restore_backup(name):
        get_logger().info("Restoring Backup...")

        try:
            path = Helper.get_model_path(name) + Settings.NAME_MODEL
            if not os.path.isfile(path):
                raise FileNotFoundError

            return load_model(path)

        except FileNotFoundError:
            get_logger().error(
                'Do not have Backup! Starting new?')
            return None

    @staticmethod
    def do_backup(worker):
        get_logger().info("Making Backup...")

        Helper.backup_model_with(
            Helper.get_model_path(
                worker.C_PARAMS.name
            ),
            worker.MODEL,
            worker.P_PARAMS,
            worker.C_PARAMS,
            worker.VISUAL
        )

    @staticmethod
    def plot_structure(worker):
        path = Helper.get_model_path(
            worker.C_PARAMS.name
        )
        plot_model(
            worker.MODEL,
            to_file=path + Settings.NAME_STRUCTURE,
            show_shapes=True,
            show_layer_names=True
        )
        return plt

    @staticmethod
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


#####################################
if __name__ == "__main__":

    def combine_workers():
        workers = [Worker(
            Worker.Params(
                '2021-New-V2',
                train_part=0.8,
            ),
            Preprocessor.Params(
                backward=(0, 1),
                func=Formats.formatting_ex2,
                augmenter=Augmenters.get_new_validation(),
            ),
        )]
        return workers


    def start_actions():
        for worker in combine_workers():
            worker.start_training()


    start_actions()
