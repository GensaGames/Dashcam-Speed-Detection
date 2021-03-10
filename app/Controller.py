import jsonpickle
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

    def __init__(self, w_params, p_params, model):
        self.W_PARAMS = w_params
        self.P_PARAMS = p_params
        self.MODEL = model

    def start_training(self):
        get_logger().info(
            'Start training from Controller!'
        )

        result = self.MODEL.fit_generator(
            steps_per_epoch=500,
            epochs=280,
            validation_steps=100,
            generator=self.__get_generator(),
            validation_data=self.__get_generator(
                validation=True
            ),
            callbacks=[
                ModelCheckpoint(
                    Helper.get_model_path(
                        self.W_PARAMS.name) + Settings.NAME_MODEL,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1,
                )
            ]
        )

        get_logger().info(
            'Data fitting is done. Return the results. '
        )
        return result

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
                    self.W_PARAMS.train_part
                )

            indexes, source, _ = data.get_train_batch() \
                if not validation else \
                data.get_validation_batch()

            get_logger().debug(
                'Gen Source: {}'.format(source.name)
            )

            aug = Augmenters.get_new_validation() if validation \
                else Augmenters.get_new_training()

            x, y = Preprocessor(self.P_PARAMS, aug) \
                .build(indexes, source.path, source.y_values) \
                .run()

            yield x, y


class Utils:

    @staticmethod
    def backup_params(name, *params):
        get_logger().info("Making Params Backup...")
        path = Helper.get_model_path(name)

        for i in params:
            with open(path + type(i).__qualname__, "w+") as file:
                file.write(jsonpickle.encode(i))

    @staticmethod
    def restore_backup(name):
        get_logger().info("Restoring Backup...")
        path = Helper.get_model_path(name)

        # noinspection PyBroadException
        try:
            params = []
            for i in [Worker.Params, Preprocessor.Params]:
                with open(path + i.__qualname__, "rb") as file:
                    params.append(
                        jsonpickle.decode(file.read())
                    )

            return Worker(*params, load_model(
                path + Settings.NAME_MODEL))

        except Exception:
            get_logger().error(
                'Do not have Backup! Starting new.')
            return None

    @staticmethod
    def plot_structure(name, model):
        get_logger().info("Plotting Model Structure...")
        path = Helper.get_model_path(name)

        plot_model(
            model,
            to_file=path + Settings.NAME_STRUCTURE,
            show_shapes=True,
            show_layer_names=True
        )
        return plt

    @staticmethod
    def plot_history(name, history):
        get_logger().info(
            "Plotting History Object: {} ..."
                .format(history.keys())
        )
        path = Helper.get_model_path(name)

        try:
            plt.figure(1)
            plt.plot(history['accuracy'])
            plt.plot(history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(path + 'Model-Accuracy.png')
        except Exception as e:
            get_logger().error(e)

        try:
            plt.figure(2)
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(path + 'Model-Loss.png')
        except Exception as e:
            get_logger().error(e)


#####################################
if __name__ == "__main__":

    def combine_workers():
        workers = [Worker(
            Worker.Params(
                '2021-New-V3',
                train_part=0.8,
            ),
            Preprocessor.Params(
                backward=(0, 1),
                func=Formats.formatting_ex2,
            ),
            Models.nvidia_model(),
        )]
        return workers


    def start_actions():
        for worker in combine_workers():
            name = worker.W_PARAMS.name

            # Try to Restore from backup
            actual = Utils.restore_backup(name) \
                     or worker

            # Check Start checkpoint
            if actual == worker:
                Utils.plot_structure(name, actual.MODEL)
                Utils.backup_params(
                    name, *[actual.W_PARAMS, actual.P_PARAMS]
                )

            result = actual.start_training()
            Utils.plot_history(name, result.history)


    start_actions()
