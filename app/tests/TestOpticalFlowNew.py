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


def opticalFlowOverlay(image_current, image_next):

    feature_params = dict( maxCorners = 500,
                           qualityLevel = 0.1,
                           minDistance = 7,
                           blockSize = 5 )
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    image_current_saved = np.copy(image_current)
    # image_next_saved = np.copy(image_next)
    image_next_saved = np.zeros_like(image_next)

    image_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    image_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    # Finds edges in an image using the [Canny86] algorithm.
    p0 = cv2.goodFeaturesToTrack(image_current, mask = None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(image_current, image_next, p0, None, **lk_params)

    color = np.random.randint(0, 255, (100, 3))

    mask = np.zeros_like(image_current)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel() # flatten
        c, d = old.ravel()
        image_next_saved = cv2.arrowedLine(mask, (a,b), (c, d), color[i%100].tolist(), 2, 8, tipLength=3)

        # image_next_saved = cv2.circle(image_next_saved, (a, b), 25, color[i%100].tolist(), -1)
        image_next_fg = cv2.bitwise_and(image_next, image_next, mask = mask)

    # dst = cv2.add(image_next, image_next_fg)
    return image_next_saved


def test_opencv_optical_moving():
    for _ in range(4000, 20400, 20):

        for i in range(_, _ + 10):
            ia = Augmenters.get_new_validation()

            image_current = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i) + '.jpg', cv2.IMREAD_COLOR)
            image_current = ia.augment_image(image_current)
            image_current = cv2.resize(
                image_current[100:-160, 80:-80], (0, 0), fx=1, fy=1)

            image_next = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_COLOR)
            image_next = ia.augment_image(image_next)
            image_next = cv2.resize(
                image_next[100:-160, 80:-80], (0, 0), fx=1, fy=1)

            img_new = opticalFlowOverlay(image_current, image_next)
            cv2.imshow('frame1', img_new)
            cv2.waitKey(0)


test_opencv_optical_moving()



