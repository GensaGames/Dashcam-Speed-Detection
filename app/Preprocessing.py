from __future__ import division

from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
import rx
from rx import operators as ops

from app.Data import Data
from app.tools import Augmenters
from app.other.Parameters import PreprocessorParams
from app.other.LoggerFactory import get_logger


class Preprocessor:

    def __init__(self, params, augmenter):
        self.PARAMS = params
        self.AUGMENTER = augmenter

    def __take_x(self, indexes, path):

        assert min(indexes) - max(
            self.PARAMS.backward) >= 0

        complete_path = sorted(Path(
            path).glob('*'), key=lambda x: float(x.stem))

        items = []
        for i in indexes:
            # Look Back Strategy for the previous
            # Y Frames from started Idx position
            looking_back = list(map(
                lambda x: i - x, self.PARAMS.backward))

            list_paths = itemgetter(
                *looking_back)(complete_path)

            img_aug = self.AUGMENTER.image.to_deterministic()

            for full_path in np.flipud(np.array([list_paths]).flatten()):

                image = cv2.imread(
                    str(full_path), cv2.IMREAD_COLOR)
                image = img_aug.augment_image(image)
                items.append(image)

        assert len(indexes) * (
            len(self.PARAMS.backward)) == len(items)
        return items

    def __take_y(self, indexes, y):
        if y is None:
            return None

        assert min(indexes) - \
               max(self.PARAMS.backward) >= 0

        y_values = []
        for i in indexes:
            y_values.append(y[i])

        y_values = np.reshape(
            y_values, (len(y_values), 1))
        return y_values

    def __map_crop(self, frames):
        assert isinstance(frames, list) and \
               isinstance(frames[0], np.ndarray)

        # Apply random floating Area Shift
        def shift():
            if self.AUGMENTER.area_float is 0:
                return 0

            return np.random.randint(
                low=-1 * self.AUGMENTER.area_float,
                high=self.AUGMENTER.area_float)

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
        # noinspection DuplicatedCode
        def get_flow_change(img1, img2):
            hsv = np.zeros_like(img1)
            # set saturation
            hsv[:, :, 1] = cv2.cvtColor(
                img2, cv2.COLOR_RGB2HSV)[:, :, 1]

            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None,
                0.6, 4, 20, 3, 5, 1.1, 0)

            # convert from cartesian to polar
            mag, ang = cv2.cartToPolar(
                flow[..., 0], flow[..., 1])

            # hue corresponds to direction
            hsv[:, :, 0] = ang * (180 / np.pi / 2)

            # value corresponds to magnitude
            hsv[:, :, 2] = cv2.normalize(
                mag, None, 0, 255, cv2.NORM_MINMAX)

            # Сonvert HSV to float32's
            new = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)

            """
            Comment/Uncomment for showing each image
            moving optical flow.
            """
            # cv2.imshow('Preprocessing Flow.', new)
            # cv2.waitKey(0)
            return new

        flow_frames = []
        for line in range(0, len(frames), timeline):

            for idx in range(line, line + timeline - 1):
                """
                Comment/Uncomment for showing each image
                moving optical flow.
                """
                # cv2.imshow('Preprocessing Flow.', frames[idx])
                # cv2.waitKey(0)
                optical = get_flow_change(
                    frames[idx],
                    frames[idx + 1]
                )
                flow_frames.append(optical)

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

        samples = int(len(
            frames) / timeline)

        return np.array(frames).reshape((
            samples,
            timeline,
            frames[0].shape[0],
            frames[0].shape[1],
            3 if len(frames[0].shape) > 2 else 1
        ))

    @staticmethod
    def __to_timeline_y(frames):
        if frames is None:
            return None
        assert isinstance(frames, np.ndarray)
        return frames.reshape(len(frames), 1)

    # noinspection PyUnresolvedReferences
    def build(self, indexes, path, y_values):

        x = self.__take_x(indexes, path)
        obs_x = rx.of(x).pipe(
            ops.map(self.__map_crop),
            ops.map(self.__map_scale),
            ops.map(self.__build_optical_flow),
            ops.map(self.__map_normalize),
            ops.map(self.__to_timeline_x),
        )

        y = self.__take_y(indexes, y_values)
        obs_y = rx.of(y).pipe(
            ops.map(self.__to_timeline_y),
        )
        return rx.zip(obs_x, obs_y)


#####################################
if __name__ == "__main__":
    logger = get_logger()

    def validate():

        def on_ready(x_y):
            logger.info('Received X shape {}'
                        .format(x_y[0].shape))
            logger.info('Received Y shape {}'
                        .format(x_y[1].shape))
            assert x_y[0].shape[0] == x_y[1].shape[0]

        values, source, _ = Data()\
            .initialize(0.7)\
            .get_train_batch(5)

        logger.debug(
            'Requesting from Source: {}. Next Indexes: {}\n\n'
                .format(source.name, values))

        Preprocessor(
            PreprocessorParams(
                backward=(0, 1, 2, 3),
                frame_y_trim=(230, -150),
                frame_x_trim=(180, -180),
                frame_scale=1.4,
            ),
            Augmenters.get_new_training()
        ).build(
            values, source.path, source.y_values
        ).subscribe(on_ready, on_error=lambda e: logger
                    .error('Exception! ' + str(e)))

    validate()
