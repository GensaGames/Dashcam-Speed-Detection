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
        self.model = None

    def start_training_epochs(self):

        for e in range(self.C_PARAMS.epochs):
            get_logger().info('Starting {} Training Epoch!'
                              .format(str(e)))
            data = Data() \
                .initialize(self.C_PARAMS.train_part)

            step = 0
            while True:
                t_idx, t_source, _ = data \
                    .get_train_batch(self.C_PARAMS.baths)

                if not t_source or not t_idx:
                    break

                get_logger().debug(
                    'Processing Training Step: {}'
                        .format(step))

                Preprocessor(
                    self.P_PARAMS,
                    Augmenters.get_new_training()
                ).build(
                    t_idx, t_source.path, t_source.y_values
                ).subscribe(
                    self.__step_model,
                    on_error=lambda error: get_logger()
                        .error('Exception! ' + str(error))
                )

                step += 1
                if step % self.C_PARAMS.step_vis == 0:
                    v_idx, v_source, _ = data\
                        .get_validation_batch(120)
                    self.__step_validation(v_idx, v_source)
                    self.do_backup()

            get_logger().debug('Epoch: {} Training Done!'
                               .format(e))

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

    def __step_validation(self, indexes, source):
        get_logger().info("Starting Cross Validation.")

        def __internal(x_y):
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
                indexes, source.path, source.y_values
            ).subscribe(
                __internal,
                on_error=lambda error: get_logger()
                    .error('Exception! ' + str(error))
            )


#####################################
if __name__ == "__main__":

    def combine_workers():
        workers = [MiniBatchWorker(
            PreprocessorParams(
                backward=(0, 1, 2, 3),
                frame_y_trim=(230, -160),
                frame_x_trim=(180, -180),
                frame_scale=1),

            ControllerParams(
                'STAR-3D-V3',
                baths=30,
                train_part=0.7,
                epochs=1,
                step_vis=10))
        ]
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
            worker.start_training_epochs()
            plot_progress(worker)


    start_actions()
