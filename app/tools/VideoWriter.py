import os

import cv2

from app import Settings
from app.Controller import MiniBatchWorker
from app.tools import Augmenters
from app.other.Parameters import PreprocessorParams, ControllerParams
from app.Preprocessing import Preprocessor
from app.other.LoggerFactory import get_logger


class VideoWriter:
    def __init__(self, params):
        self.PARAMS = params

        self.worker = MiniBatchWorker(
            PreprocessorParams(),
            ControllerParams(
                'NEW-OPT-FIN-2'
            )
        )
        self.worker.restore_backup()

        self.preproc = Preprocessor(
            PreprocessorParams(
                backward=(0, 1), frame_y_trim=(110, -160),
                frame_x_trim=(10, -130), frame_scale=1.4,
            ),
            Augmenters.get_new_validation()
        )

    def process(self, image1, image2, memory):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        x = self.preproc.buildOne(
            image1, image2).run()

        y = self.worker.MODEL.predict(x)

        # Smooth Aggressive:
        # if len(memory) > 10:
        #     previous = memory[-20:]
        #     avg = sum(previous) / len(previous)
        #     if abs(y - avg) > 5:
        #         y = avg
        #     else:
        #         y = (sum(previous) + y) / (len(previous) + 1)

        memory.append(y)
        get_logger().info(y)

        """
        Comment/Uncomment for showing each image
        moving optical flow.
        """
        # cv2.imshow('Video Image', cv2.resize(
        #     image1[110:-160, 0:-120], (0, 0), fx=1.4, fy=1.4))
        # cv2.waitKey(0)

        cv2.imshow('Video Image', image2)
        cv2.waitKey(0)

    def start(self):
        if not os.path.exists(self.PARAMS.path):
            get_logger().error('Cannot find File: {}'
                         .format(self.PARAMS.path))
            return

        vid = cv2.VideoCapture(self.PARAMS.path)
        memory = []

        pvFrame, index = None, 0
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            if pvFrame is not None:
                self.process(pvFrame, frame, memory)
            pvFrame = frame
            index += 1


class VideoParams:
    def __init__(self, path):
        self.path = path


#####################################
if __name__ == "__main__":
    VideoWriter(VideoParams(
        Settings.WRITE_VIDEO_2)
    ).start()
