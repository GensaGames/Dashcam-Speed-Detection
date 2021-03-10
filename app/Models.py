from keras import Sequential
from keras.activations import linear
from keras.engine import InputLayer
from keras.initializers import he_normal
from keras.layers import Dense, ELU, Flatten, Conv2D, Conv3D, BatchNormalization, MaxPooling3D, Lambda, Dropout
from keras.losses import mean_squared_error
from keras.optimizers import Adam


class Models:

    @staticmethod
    def get2D_CNN(x):
        x_shape = x.shape
        input_shape = (
            x_shape[1],
            x_shape[2],
            x_shape[3])

        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(
            filters=24, kernel_size=(5, 5), strides=(2, 2),
            padding='valid',
            kernel_initializer=he_normal(),
        ))
        model.add(ELU())

        model.add(Conv2D(
            filters=36, kernel_size=(5, 5), strides=(2, 2),
            padding='valid',
            kernel_initializer=he_normal(),
        ))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Conv2D(
            filters=48, kernel_size=(5, 5), strides=(2, 2),
            padding='valid',
            kernel_initializer=he_normal(),
        ))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Conv2D(
            filters=48, kernel_size=(3, 3), strides=(1, 1),
            padding='valid',
            kernel_initializer=he_normal(),
        ))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Conv2D(
            filters=64, kernel_size=(3, 3), strides=(1, 1),
            padding='valid',
            kernel_initializer=he_normal(),
        ))
        model.add(ELU())
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
            .add(Dense(units=1,
                       kernel_initializer=he_normal(),
                       activation=linear))
        model \
            .compile(loss=mean_squared_error,
                     optimizer=Adam(lr=1e-4))
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
        model.add(Conv3D(
            filters=86, kernel_size=(3, 5, 5), strides=(1, 2, 2),
            padding='same',
            kernel_initializer=he_normal(),
            input_shape=input_shape
        ))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Conv3D(
            filters=64, kernel_size=(2, 3, 3), strides=(1, 2, 2),
            padding='valid',
            kernel_initializer=he_normal(),
        ))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Conv3D(
            filters=64, kernel_size=(2, 3, 3), strides=(1, 1, 1),
            padding='valid',
            kernel_initializer=he_normal(),
        ))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Conv3D(
            filters=64, kernel_size=(2, 3, 3), strides=(1, 1, 1),
            padding='valid',
            kernel_initializer=he_normal(),
        ))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Conv3D(
            filters=48, kernel_size=(2, 3, 3), strides=(1, 1, 1),
            padding='valid',
            kernel_initializer=he_normal(),
        ))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(
            MaxPooling3D(
                pool_size=(1, 2, 2),
                padding="valid"
            )
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
            .add(Dense(units=1,
                       kernel_initializer=he_normal(),
                       activation=linear))
        model.compile(
            loss=mean_squared_error,
            optimizer=Adam(lr=1e-4),
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def nvidia_model():
        N_img_height = 66
        N_img_width = 220
        N_img_channels = 3
        inputShape = (N_img_height, N_img_width, N_img_channels)

        model = Sequential()
        # normalization
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=inputShape))

        model.add(Conv2D(24, 5, 5,
                         subsample=(2, 2),
                         border_mode='valid',
                         init='he_normal',
                         name='conv1'))

        model.add(ELU())
        model.add(Conv2D(36, 5, 5,
                         subsample=(2, 2),
                         border_mode='valid',
                         init='he_normal',
                         name='conv2'))

        model.add(ELU())
        model.add(Conv2D(48, 5, 5,
                         subsample=(2, 2),
                         border_mode='valid',
                         init='he_normal',
                         name='conv3'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Conv2D(64, 3, 3,
                         subsample=(1, 1),
                         border_mode='valid',
                         init='he_normal',  # gaussian init
                         name='conv4'))

        model.add(ELU())
        model.add(Conv2D(64, 3, 3,
                         subsample=(1, 1),
                         border_mode='valid',
                         init='he_normal',
                         name='conv5'))

        model.add(Flatten(name='flatten'))
        model.add(ELU())
        model \
            .add(Dense(units=100,
                       kernel_initializer=he_normal(),
                       activation=linear))
        model.add(ELU())
        model \
            .add(Dense(units=50,
                       kernel_initializer=he_normal(),
                       activation=linear))
        model.add(ELU())
        model \
            .add(Dense(units=10,
                       kernel_initializer=he_normal(),
                       activation=linear))
        model.add(ELU())

        model \
            .add(Dense(units=1,
                       kernel_initializer=he_normal(),
                       activation=linear))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(
            optimizer=adam,
            loss='mse',
            metrics=['accuracy']
        )

        return model
