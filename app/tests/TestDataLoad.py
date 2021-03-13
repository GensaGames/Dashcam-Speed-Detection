import os

import cv2
import numpy as np
from app import Settings
from app.Controller import TrainingData


def iterate_custom():
    path = Settings.PROJECT + 'custom/Chunk_1/'
    print(path)

    for d1 in os.listdir(path):
        for d2 in os.listdir(path + d1):
            outer = os.path.join(path, d1, d2)
            print('Outer Path: {}'
                  .format(outer))

            y_file = os.path.join(
                outer,
                'processed_log/CAN/speed/value'
            )
            assert os.path.isfile(y_file)
            print('Y Records: {}'
                  .format(len(np.fromfile(y_file))))

            video_to_frame(
                os.path.join(outer, 'video.hevc'),
                os.path.join(outer, 'frames/'),
            )


def video_to_frame(path_from, path_to, img_format='.jpg'):
    print('Start video Recording!')
    vid = cv2.VideoCapture(path_from)
    if not os.path.exists(path_to):
        os.makedirs(path_to)

    index = 0
    while True:
        # Extract images
        ret, frame = vid.read()
        if not ret:
            break

        # Saves images
        name = path_to + str(index) + img_format

        # Transform from 1164 x 874
        cv2.imwrite(name, cv2.resize(
            frame, (640, 480)))
        # next frame
        index += 1


iterate_custom()
