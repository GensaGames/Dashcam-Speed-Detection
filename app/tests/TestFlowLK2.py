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


def opticalFlowOverlay(imagePv, image):
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=5)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    imagePv = cv2.cvtColor(imagePv, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Finds edges in an image using the [Canny86] algorithm.
    p0 = cv2.goodFeaturesToTrack(imagePv, mask=None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(imagePv, image, p0, None, **lk_params)
    color = [255, 255, 255]

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    mask = np.zeros_like(image)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        distance = math.hypot(a-c, b-d)

        if distance < 1.0:
            color = [100, 100, 100]
        else:
            color = [255, 255, 255]

        mask = cv2.arrowedLine(
            mask, (a, b), (c, d), color,
            2, 1, tipLength=3)
    return mask


def opticalFlowOverlay2(imagePv, image):
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=5)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    imagePvGray = cv2.cvtColor(imagePv, cv2.COLOR_RGB2GRAY)
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Finds edges in an image using the [Canny86] algorithm.
    p0 = cv2.goodFeaturesToTrack(imageGray, mask=None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        imagePvGray, imageGray, p0, None, **lk_params)
    color = np.random.randint(0, 255, (300, 3))

    colorL = np.array([122, 122, 122])
    colorC = np.array([255, 255, 255])

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    mask = np.zeros_like(image)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        mask = cv2.line(mask, (a, b), (c, d), colorL.tolist(), 1)
        image = cv2.circle(mask, (a, b), 2, colorC.tolist(), -1)

    return cv2.add(image, mask)


def testOpenCVOpticalMoving():
    for _ in range(100, 20400, 20):

        for i in range(_, _ + 10):
            ia = Augmenters.get_new_validation()

            image_current = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i) + '.jpg', cv2.IMREAD_COLOR)
            image_current = ia.augment_image(image_current)
            image_current = cv2.resize(
                image_current[80:-160, 60:-60], (0, 0), fx=1, fy=1)

            image_next = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_COLOR)
            image_next = ia.augment_image(image_next)
            image_next = cv2.resize(
                image_next[80:-160, 60:-60], (0, 0), fx=1, fy=1)

            img_new = opticalFlowOverlay2(image_current, image_next)
            cv2.imshow('frame1', img_new)
            cv2.waitKey(0)


testOpenCVOpticalMoving()
