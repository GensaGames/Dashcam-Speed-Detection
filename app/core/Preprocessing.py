from __future__ import division

import logging
from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
from numpy import loadtxt
from rx import Observable

from app import Settings
from app.core.Augmenters import AugmenterModel
from app.core.Parameters import PreprocessorParams

logging.basicConfig(level=logging.INFO)


class Preprocessor:

    def __init__(self, params, augmenter):
        self.PARAMS = params
        self.AUGMETER = augmenter

    def __load_y(self, path_indexes):
        assert min(path_indexes[1]) - \
               max(self.PARAMS.backward) >= 0

        items = loadtxt(
            path_indexes[0], delimiter=" ",
            unpack=False)[path_indexes[1]]

        items = np.reshape(
            items, (len(items), 1))
        return items

    def __load_x(self, path_indexes):
        assert min(path_indexes[1]) - \
               max(self.PARAMS.backward) >= 0

        complete_path = sorted(Path(
            path_indexes[0]).glob('*'), key=lambda x: float(x.stem))

        items = []
        for i in path_indexes[1]:

            # Look Back Strategy for the previous
            # Y Frames from started Idx position
            looking_back = list(map(
                lambda x: i - x, self.PARAMS.backward))

            state = self.AUGMETER.model

            list_paths = itemgetter(*looking_back)(complete_path)
            for path in np.flipud(np.array([list_paths]).flatten()):

                image = cv2.imread(
                    str(path), cv2.IMREAD_GRAYSCALE)
                items.append(state.augment_image(image))

        assert len(path_indexes[1]) * (
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

            for idx in range (timeline, timeline + step):
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
            frames[idx] = val / 256.0

        frames = np.array(frames)
        return frames

    # Merging previous Frames to the Timeline
    # With 3D array of Samples * Timeline * Features
    def __to_timeline_x(self, frames):
        assert isinstance(frames, np.ndarray)

        timeline = len(self.PARAMS.backward)
        assert len(frames) % timeline == 0
        len_indexes = int(len(frames) / timeline)

        return frames.reshape((
            len_indexes, timeline, frames[0].shape[0],
            frames[0].shape[1], 1))

    @staticmethod
    def __to_timeline_y(frames):
        assert isinstance(frames, np.ndarray)
        return frames.reshape(len(frames), 1)

    # noinspection PyUnresolvedReferences
    def build(self, path_x, path_y, indexes):

        obs_x = Observable.of((path_x, indexes)) \
            .map(self.__load_x) \
            .map(self.__map_crop) \
            .map(self.__map_scale) \
            .map(self.__map_normalize) \
            .map(self.__to_timeline_x)

        obs_y = Observable.of((path_y, indexes)) \
            .map(self.__load_y) \
            .map(self.__to_timeline_y)

        return Observable.zip(
            obs_x, obs_y, lambda x, y: (x, y))


#####################################
if __name__ == "__main__":
    def __assert(x_y):
        logging.info('X shape {}'.format(x_y[0].shape))
        logging.info('Y shape {}'.format(x_y[1].shape))

        assert x_y[0].ndim == 5 and x_y[1].ndim == 2 \
               and x_y[0].shape[0] == x_y[1].shape[0]


    Preprocessor(PreprocessorParams(
        (0, 1, 2), frame_scale=1.5), AugmenterModel()).build(
        '../../' + Settings.TRAIN_FRAMES,
        '../../' + Settings.TRAIN_Y, [2, 10, 86]) \
        .subscribe(__assert)
