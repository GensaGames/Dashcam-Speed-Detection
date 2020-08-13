import matplotlib.pyplot as plt
from keras.utils import plot_model

import app.other.Helper as Helper
from app import Settings
from app.Data import Data
from app.tools import Augmenters
from app.Models import Models
from app.other.Parameters import ControllerParams, \
    VisualHolder, PreprocessorParams
from app.Preprocessing import Preprocessor
from app.other.LoggerFactory import get_logger


class MiniBatchWorker:

    def __init__(self, p_params, c_params):
        self.P_PARAMS, self.C_PARAMS, self.VISUAL \
            = p_params, c_params, VisualHolder()
        self.MODEL = None

    def start_training(self):
        # Iterate over Controller Epochs
        for e in range(self.C_PARAMS.epochs):
            get_logger().info('Starting {} Training Epoch!'
                              .format(str(e)))

            data = Data().initialize(
                len(self.P_PARAMS.backward),
                self.C_PARAMS.train_part
            )

            step = 0
            # Iterate over all known Samples
            while True:
                t_idx, t_source, _ = data \
                    .get_train_batch(self.C_PARAMS.baths)

                if not t_source or not len(t_idx):
                    get_logger().debug(
                        'Epoch: {} Training Done!'.format(e))
                    break

                # Batch Training Step
                step += 1
                get_logger().debug(
                    'Training Process. Source: {} Step: {}'
                        .format(t_source.name, step))

                self.__step_model(t_idx, t_source)

                if step % self.C_PARAMS.step_vis == 0:
                    # Validation Step and Visualization
                    v_idx, v_source, _ = data\
                        .get_validation_batch(120)
                    self.__step_validation(v_idx, v_source)
                    WorkerUtils.do_backup(self)

    def __step_model(self, indexes, source):

        def __internal(x_y):
            if self.MODEL is None:
                self.MODEL = Models.get3D_CNN(x_y[0])
                WorkerUtils.plot_structure(self)

            value = self.MODEL.train_on_batch(x_y[0], x_y[1])
            self.VISUAL.add_training_point(value)

            get_logger().debug('Training Batch loss: {}'
                               .format(value))
        Preprocessor(
            self.P_PARAMS,
            Augmenters.get_new_training()
        ).build(
            indexes, source.path, source.y_values
        ).subscribe(
            __internal,
            on_error=lambda error: get_logger()
                .error('Exception! ' + str(error))
        )

    def __step_validation(self, indexes, source):
        get_logger().debug("Starting Cross Validation. Source: {}"
                           .format(source.name))

        def __internal(x_y):
            mse = self.MODEL \
                .evaluate(x_y[0], x_y[1])

            get_logger().debug(
                "Cross Validation Done on Items Size: {} "
                "MSE: {}".format(len(x_y[0]), mse))

            self.VISUAL.add_validation_point(mse)

        Preprocessor(
            self.P_PARAMS,
            Augmenters.get_new_validation()
        ).build(
            indexes, source.path, source.y_values
        ).subscribe(
            __internal,
            on_error=lambda error: get_logger()
                .error('Exception! ' + str(error))
        )


class WorkerUtils:

    @staticmethod
    def restore_backup(worker):
        get_logger().info("Restoring Backup...")

        if worker.MODEL is not None:
            get_logger().error(
                'Model already created. Do not override!')
            return

        try:
            path = Helper.get_model_path(
                Settings.BUILD,
                worker.C_PARAMS.name
            )
            worker.MODEL, worker.P_PARAMS, worker.C_PARAMS, worker.VISUAL =\
                Helper.restore_model_with(path)
        except FileNotFoundError:
            get_logger().error(
                'Do not have Backup! Starting new.')

    @staticmethod
    def do_backup(worker):
        get_logger().info("Making Backup...")

        path = Helper.get_model_path(
            Settings.BUILD,
            worker.C_PARAMS.name
        )

        Helper.backup_model_with(
            path,
            worker.MODEL,
            worker.P_PARAMS,
            worker.C_PARAMS,
            worker.VISUAL
        )

    @staticmethod
    def plot_structure(worker):
        path = Helper.get_model_path(
            Settings.BUILD,
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
        workers = [MiniBatchWorker(
            PreprocessorParams(
                backward=(0, 1, 2, 3),
                frame_y_trim=(230, -160),
                frame_x_trim=(180, -180),
                frame_scale=1.4),

            ControllerParams(
                'COMP_NEW_V1',
                baths=30,
                train_part=0.7,
                epochs=1,
                step_vis=10))
        ]
        return workers

    def start_actions():
        for worker in combine_workers():
            WorkerUtils.restore_backup(worker)
            # worker.start_training()
            WorkerUtils.plot_progress(worker)


    start_actions()
