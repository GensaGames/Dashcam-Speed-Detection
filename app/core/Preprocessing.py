from __future__ import division

import logging
from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
from numpy import loadtxt
from rx import Observable

from app import Settings
from app.core import Augmenters
from app.core.Parameters import PreprocessorParams


class Preprocessor:

    def __init__(self, params, augmenter=None):
        self.PARAMS = params
        self.AUGMETER = augmenter

    def __load_y(self, path_indexes):
        if path_indexes[0] is None:
            return None
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

            list_paths = itemgetter(*looking_back)(complete_path)
            for path in np.flipud(np.array([list_paths]).flatten()):

                image = cv2.imread(
                    str(path), cv2.IMREAD_COLOR)
                image = self.AUGMETER.augment_image(image) if \
                    self.AUGMETER is not None else image
                items.append(image)

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

    def __build_optical_flow(self, frames):

        timeline = len(self.PARAMS.backward)
        assert len(frames) % timeline == 0

        # Main function for optical flow detection
        def get_flow_change(img1, img2):
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY), None,
                0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(
                flow[..., 0], flow[..., 1])

            hsv = np.zeros_like(img1)
            hsv[:, :, 1] = cv2.cvtColor(
                img2, cv2.COLOR_RGB2HSV)[:, :, 1]

            hsv[..., 0] = ang * (180 / np.pi / 2)
            hsv[..., 2] = cv2.normalize(
                mag, None, 0, 255, cv2.NORM_MINMAX)

            hsv = np.asarray(hsv, dtype=np.float32)

            # Comment/Uncomment for showing each image
            # moving optical flow.

            # cv2.imshow('Original', cv2.cvtColor(
            #     hsv, cv2.COLOR_HSV2BGR))
            # cv2.waitKey(0)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        flow_frames = []
        for line in range(0, len(frames), timeline):

            for idx in range(line, line + timeline - 1):
                image_current = frames[idx]
                image_next = frames[idx + 1]

                flow_frames.append(get_flow_change(
                    image_current, image_next))

        return flow_frames

    @staticmethod
    def __map_normalize(frames):
        assert isinstance(frames, list) and \
               isinstance(frames[0], np.ndarray)

        for idx, val in enumerate(frames):
            frames[idx] = val / 256.0

        return frames

    # Merging previous Frames to the Timeline
    # With 3D array of Samples * Timeline * Features
    def __to_timeline_x(self, frames):
        timeline = len(self.PARAMS.backward) -1
        assert len(frames) % timeline == 0

        delta_len = int(len(
            frames) / timeline)

        return np.array(frames).reshape((
            delta_len, timeline, frames[0].shape[0],
            frames[0].shape[1], 3))

    @staticmethod
    def __to_timeline_y(frames):
        if frames is None:
            return None
        assert isinstance(frames, np.ndarray)
        return frames.reshape(len(frames), 1)

    # noinspection PyUnresolvedReferences
    def build(self, path_x, path_y, indexes):

        obs_x = Observable.of((path_x, indexes)) \
            .map(self.__load_x) \
            .map(self.__map_crop) \
            .map(self.__map_scale) \
            .map(self.__build_optical_flow) \
            .map(self.__to_timeline_x)

        obs_y = Observable.of((path_y, indexes)) \
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

        assert x_y[0].ndim == 5 and x_y[1].ndim == 2 \
               and x_y[0].shape[0] == x_y[1].shape[0]


    Preprocessor(PreprocessorParams(
        (0, 1, 2), frame_scale=1.5, frame_x_trim=(0, 640),
        frame_y_trim=(0, 480), area_float=0),

        Augmenters.get_new_validation()).build(
        '../../' + Settings.TRAIN_FRAMES,
        '../../' + Settings.TRAIN_Y, [10, 86, 170]) \
        .subscribe(__assert)
