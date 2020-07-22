from keras import Sequential
from keras.activations import linear
from keras.initializers import he_normal
from keras.layers import Dense, ELU, Flatten, Conv2D, Conv3D, BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam


class Models:

    @staticmethod
    def get2D_CNN(x):
        input_shape = (
            x.shape[1],
            x.shape[2],
            x.shape[3],
        )
    
        model = Sequential()
        model.add(
            Conv2D(filters=64, kernel_size=(5, 5), strides=(3, 3),
                   input_shape=input_shape, padding='valid',
                   kernel_initializer=he_normal())
        )
        model.add(
            Conv2D(filters=86, kernel_size=(5, 5), strides=(2, 2),
                   padding='valid', kernel_initializer=he_normal())
        )
        model.add(
            Conv2D(filters=86, kernel_size=(3, 3), strides=(2, 2),
                   padding='valid', kernel_initializer=he_normal())
        )
        model.add(
            Conv2D(filters=86, kernel_size=(3, 3), strides=(1, 1),
                   padding='valid', kernel_initializer=he_normal())
        )
        model.add(Flatten())
        model \
            .add(Dense(units=256,
                       kernel_initializer=he_normal()))
        model.add(ELU())
    
        model \
            .add(Dense(units=128,
                       kernel_initializer=he_normal()))
        model.add(ELU())
    
        model \
            .add(Dense(units=64,
                       kernel_initializer=he_normal()))
        model.add(ELU())
    
        model \
            .add(Dense(units=1,
                       kernel_initializer=he_normal(),
                       activation=linear))
        model \
            .compile(loss=mean_squared_error,
                     optimizer=Adam())
        return model

    @staticmethod
    def get3D_CNN(x):
        x_shape = x.shape
        input_shape = (
            x_shape[1],
            x_shape[2],
            x_shape[3],
            x_shape[4])

        model = Sequential()
        model.add(
            Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(2, 2, 2),
                   input_shape=input_shape, padding='same',
                   kernel_initializer=he_normal()))
        model.add(BatchNormalization())

        model.add(
            Conv3D(filters=64, kernel_size=(2, 5, 5), strides=(1, 2, 2),
                   input_shape=input_shape, padding='valid',
                   kernel_initializer=he_normal()))
        model.add(BatchNormalization())

        model.add(
            Conv3D(filters=64, kernel_size=(2, 3, 3), strides=(1, 1, 1),
                   input_shape=input_shape, padding='valid',
                   kernel_initializer=he_normal()))
        model.add(BatchNormalization())

        model.add(
            Conv3D(filters=86, kernel_size=(1, 3, 3), strides=(1, 1, 1),
                   input_shape=input_shape, padding='valid',
                   kernel_initializer=he_normal()))
        model.add(BatchNormalization())
        model.add(Flatten())

        model \
            .add(Dense(units=256,
                       kernel_initializer=he_normal()))
        model.add(ELU())

        model \
            .add(Dense(units=128,
                       kernel_initializer=he_normal()))
        model.add(ELU())

        model \
            .add(Dense(units=64,
                       kernel_initializer=he_normal()))
        model.add(ELU())

        model \
            .add(Dense(units=1,
                       kernel_initializer=he_normal(),
                       activation=linear))
        model \
            .compile(loss=mean_squared_error,
                     optimizer=Adam(lr=1e-4))
        return model
