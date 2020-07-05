from __future__ import division

from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
import rx
from numpy import loadtxt
from rx import operators as ops

from app import Settings
from app.core import Augmenters
from app.core.Parameters import PreprocessorParams
from app.other import Helper
from app.other.LoggerFactory import get_logger


class Preprocessor:

    def __init__(self, params, augmenter=None):
        self.PARAMS = params
        self.AUGMENTER = augmenter
        self.SOURCE_X_Y = None

    def __load_y(self, indexes):
        train_y_paths = self.SOURCE_X_Y[1]
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

        # Checking if indexes related to the car stop
        # variance indexes, otherwise - default
        if Helper.is_car_stop_variance(looking_back[0]):
            source_stop_frames = Settings.TRAIN_FRAMES_STOP

            # Just use folder as prefix for index, and avoid
            # collision with initial train indexes from train part
            folder = str(looking_back[0])[0]

            new_back = list(map(lambda x: x - (
                    int(folder) * Settings.PREFIX_STOP_SIZE),
                                looking_back))

            new_path = sorted(Path(
                source_stop_frames + '/' + folder).glob('*'),
                              key=lambda x: float(x.stem))
            return itemgetter(*new_back)(new_path)

        return itemgetter(*looking_back)(path)

    def __load_x(self, indexes):
        frames_paths = self.SOURCE_X_Y[0]

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
            augmenter = self.AUGMENTER.to_deterministic()

            for path in np.flipud(np.array([list_paths]).flatten()):
                image = cv2.imread(
                    str(path), cv2.IMREAD_GRAYSCALE)
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

    FLOW_FEATUES = dict(
        maxCorners=500,
        qualityLevel=0.1,
        minDistance=7,
        blockSize=5
    )

    FLOW_LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS
                  | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    def __build_optical_flow(self, frames):

        timeline = len(self.PARAMS.backward)
        assert len(frames) % timeline == 0

        # Main function for optical flow detection
        # noinspection DuplicatedCode
        def get_flow_change(img1, img2):

            # Finds edges in an image using the [Canny86] algorithm.
            p0 = cv2.goodFeaturesToTrack(
                img2, mask=None, **Preprocessor.FLOW_FEATUES
            )

            p1, st, err = cv2.calcOpticalFlowPyrLK(
                img1, img2, p0, None,
                **Preprocessor.FLOW_LK_PARAMS
            )

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            mask = np.zeros_like(img2)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                mask = cv2.line(mask, (a, b), (c, d), [122, 122, 122], 1)
                img2 = cv2.circle(mask, (a, b), 2, [255, 255, 255], -1)

            new = cv2.add(img2, mask)

            """
            Comment/Uncomment for showing each image
            moving optical flow.
            """
            # cv2.imshow('Preprocessing Flow', new)
            # cv2.waitKey(0)
            return new

        flow_frames = []
        for line in range(0, len(frames), timeline):

            for idx in range(line, line + timeline - 1):
                image = frames[idx]
                image_next = frames[idx + 1]

                flow_frames.append(get_flow_change(
                    image, image_next))

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
        timeline = len(self.PARAMS.backward) - 1
        assert len(frames) % timeline == 0

        delta_len = int(len(
            frames) / timeline)

        return np.array(frames).reshape((
            delta_len,
            frames[0].shape[0],
            frames[0].shape[1],
            1
        ))

    @staticmethod
    def __to_timeline_y(frames):
        if frames is None:
            return None
        assert isinstance(frames, np.ndarray)
        return frames.reshape(len(frames), 1)

    def set_source(self, x, y):
        self.SOURCE_X_Y = (x, y)
        return self

    # noinspection PyUnresolvedReferences
    def build(self, indexes):
        obs_x = rx.of(indexes).pipe(
            ops.map(self.__load_x),
            ops.map(self.__map_crop),
            ops.map(self.__map_scale),
            ops.map(self.__build_optical_flow),
            ops.map(self.__map_normalize),
            ops.map(self.__to_timeline_x),
        )

        obs_y = rx.of(indexes).pipe(
            ops.map(self.__load_y),
            ops.map(self.__to_timeline_y),
        )

        return rx.zip(obs_x, obs_y)


#####################################
if __name__ == "__main__":
    logger = get_logger()


    def __assert(x_y):
        logger.info('X shape {}'.format(x_y[0].shape))
        logger.info('Y shape {}'.format(x_y[1].shape))

        assert x_y[0].ndim == 4 and x_y[1].ndim == 2 \
               and x_y[0].shape[0] == x_y[1].shape[0]

    Preprocessor(PreprocessorParams(
        (0, 1), frame_scale=1.4, frame_x_trim=(70, -70),
        frame_y_trim=(100, -170), area_float=0),
        Augmenters.get_new_validation()) \
        .set_source(Settings.TEST_FRAMES, Settings.TRAIN_Y) \
        .build(list(range(67, 200))) \
        .subscribe(__assert, on_error=lambda e:
    logger.error('Exception! ' + str(e)))
