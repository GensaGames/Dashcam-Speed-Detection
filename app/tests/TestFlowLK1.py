from __future__ import division

import cv2
import numpy as np

from app.core import Augmenters
from app.other.Helper import *
from numpy import loadtxt
import cv2
import math
import app.Settings as Settings
import matplotlib.pyplot as plt


def testOpenCVOpticalMoving():
    ia = Augmenters.get_new_validation()
    for part in range(4000, 20400, 20):

        old_frame = cv2.imread(
            Settings.TEST_FRAMES + '/'
            + str(part) + '.jpg', cv2.IMREAD_COLOR)

        old_frame = ia.augment_image(old_frame)
        old_frame = cv2.resize(
            old_frame[100:-160, 80:-80], (0, 0), fx=1, fy=1)
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=500,
                              qualityLevel=0.1,
                              minDistance=7,
                              blockSize=5)
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        for i in range(part, part + 10):

            frame = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_COLOR)
            frame = ia.augment_image(frame)
            frame = cv2.resize(
                frame[100:-160, 80:-80], (0, 0), fx=1, fy=1)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # draw the tracks
            for i2, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i2].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i2].tolist(), -1)
            img = cv2.add(frame, mask)

            cv2.imshow('frame1', img)
            cv2.waitKey(0)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)


testOpenCVOpticalMoving()



