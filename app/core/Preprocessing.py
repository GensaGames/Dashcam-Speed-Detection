from __future__ import division

import logging
from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
from numpy import loadtxt
from rx import Observable
import os

from app import Settings
from app.core import Augmenters
from app.core.Parameters import PreprocessorParams
from app.other import Helper


class Preprocessor:

    def __init__(self, params, augmenter=None):
        self.PARAMS = params
        self.AUGMETER = augmenter
        self.source_x_y = None

    def __load_y(self, indexes):
        train_y_paths = self.source_x_y[1]
        if train_y_paths is None:
            return None

        assert min(indexes) - \
               max(self.PARAMS.backward) >= 0

        train_y_values = loadtxt(
            train_y_paths, delimiter=" ",
            unpack=False)

        # Get correct train Y values, in case it's
        # in car stop variance, set just 0.
        y_values = []
        for i in indexes:
            if Helper.is_car_stop_variance(i):
                y_values.append(0)
            else:
                y_values.append(train_y_values[i])

        y_values = np.reshape(
            y_values, (len(y_values), 1))
        return y_values

    @staticmethod
    def __get_index_paths(looking_back, path):
        from app.Controller import MiniBatchWorker

        # Checking if indexes related to the car stop
        # variance indexes, otherwise - default
        if Helper.is_car_stop_variance(looking_back[0]):
            source_stop_frames = Settings.TRAIN_FRAMES_STOP

            # Just use folder as prefix for index, and avoid
            # collision with initial train indexes from train part
            folder = str(looking_back[0])[0]

            new_back = list(map(lambda x: x - (
                    int(folder) * MiniBatchWorker.PREFIX_STOP_SIZE),
                                looking_back))

            new_path = sorted(Path(
                source_stop_frames + '/' + folder).glob('*'),
                              key=lambda x: float(x.stem))
            return itemgetter(*new_back)(new_path)

        return itemgetter(*looking_back)(path)

    def __load_x(self, indexes):
        frames_paths = self.source_x_y[0]

        assert min(indexes) - max(
            self.PARAMS.backward) >= 0

        complete_path = sorted(Path(
            frames_paths).glob('*'), key=lambda x: float(x.stem))

        items = []
        for i in indexes:

            # Look Back Strategy for the previous
            # Y Frames from started Idx position
            looking_back = list(map(
                lambda x: i - x, self.PARAMS.backward))

            list_paths = self.__get_index_paths(
                looking_back, complete_path)
            augmenter = self.AUGMETER.to_deterministic()

            for path in np.flipud(np.array([list_paths]).flatten()):

                image = cv2.imread(
                    str(path), cv2.IMREAD_COLOR)
                image = augmenter.augment_image(image)
                items.append(image)

        assert len(indexes) * (
            len(self.PARAMS.backward)) == len(items)
        return items

    def __map_crop(self, frames):
        assert isinstance(frames, list) and \
               isinstance(frames[0], np.ndarray)

        # Apply random floating Area Shift
        def shift():
            if self.PARAMS.area_float is 0:
                return 0

            return np.random.randint(
                low=-1 * self.PARAMS.area_float,
                high=self.PARAMS.area_float)

        step = len(self.PARAMS.backward)
        assert len(frames) % step == 0

        for timeline in range(0, len(frames), step):
            x_shift = shift()
            y_shift = shift()

            for idx in range(timeline, timeline + step):
                frames[idx] = \
                    frames[idx][
                    self.PARAMS.frame_y_trim[0] + y_shift:
                    self.PARAMS.frame_y_trim[1] + y_shift,

                    self.PARAMS.frame_x_trim[0] + x_shift:
                    self.PARAMS.frame_x_trim[1] + x_shift]
        return frames

    def __map_scale(self, frames):
        assert isinstance(frames, list) and \
               isinstance(frames[0], np.ndarray)

        if self.PARAMS.frame_scaler == 1.0:
            return frames

        for idx, frame in enumerate(frames):
            frames[idx] = cv2.resize(
                frame, (0, 0), fx=self.PARAMS.frame_scaler,
                fy=self.PARAMS.frame_scaler)
        return frames

    @staticmethod
    def __map_normalize(frames):
        assert isinstance(frames, list) and \
               isinstance(frames[0], np.ndarray)

        for idx, val in enumerate(frames):
            frames[idx] = (frames[idx].mean(axis=2) / 256.0).flatten()


        return frames

    # Merging previous Frames to the Timeline
    # With 3D array of Samples * Timeline * Features
    def __to_timeline_x(self, frames):
        timeline = len(self.PARAMS.backward)
        assert len(frames) % timeline == 0

        delta_len = int(len(
            frames) / timeline)

        return np.array(frames).reshape((
            delta_len, timeline, frames[0].shape[0]))

    @staticmethod
    def __to_timeline_y(frames):
        if frames is None:
            return None
        assert isinstance(frames, np.ndarray)
        return frames.reshape(len(frames), 1)

    def set_source(self, x, y):
        self.source_x_y = (x, y)
        return self

    # noinspection PyUnresolvedReferences
    def build(self, indexes):

        obs_x = Observable.of(indexes) \
            .map(self.__load_x) \
            .map(self.__map_crop) \
            .map(self.__map_scale) \
            .map(self.__map_normalize) \
            .map(self.__to_timeline_x)

        obs_y = Observable.of(indexes) \
            .map(self.__load_y) \
            .map(self.__to_timeline_y)

        return Observable.zip(
            obs_x, obs_y, lambda x, y: (x, y))


#####################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def __assert(x_y):
        logging.info('X shape {}'.format(x_y[0].shape))
        logging.info('Y shape {}'.format(x_y[1].shape))

        assert x_y[0].ndim == 3 and x_y[1].ndim == 2 \
               and x_y[0].shape[0] == x_y[1].shape[0]

    Preprocessor(PreprocessorParams(
        (0, 1, 2), frame_scale=1, frame_y_trim=(300, -115),
        frame_x_trim=(80, -80), area_float=0),
        Augmenters.get_new_training()) \
        .set_source(Settings.TRAIN_FRAMES, Settings.TRAIN_Y) \
        .build([18, 100006, 23]) \
        .subscribe(__assert)
