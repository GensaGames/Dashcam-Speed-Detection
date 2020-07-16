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


def opticalFlowOverlay1(image_pv, image):
    image_pv = cv2.convertScaleAbs(
        image_pv, alpha=1.5, beta=50)

    image = cv2.convertScaleAbs(
        image, alpha=1.5, beta=50)

    optical = calcOptical(image_pv, image)

    cv2.imshow('frame1', optical)
    cv2.waitKey(0)


def calcOptical(img1, img2):

    # Finds edges in an image using the [Canny86] algorithm.
    p0 = cv2.goodFeaturesToTrack(
        img2, mask=None, **dict(
            maxCorners=500,
            qualityLevel=0.1,
            minDistance=7,
            blockSize=5
        )
    )

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        img1, img2, p0, None,
        **dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS
                      | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
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
    return new


def testOpenCVOpticalMoving():
    def format_image(img):
        img = cv2.resize(
            img[250:-160, 150:-150], (0, 0), fx=2, fy=2)
        return img

    for _ in range(8250, 20400, 10):

        for i in range(_, _ + 10):
            image_pv = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
            image_pv = format_image(image_pv)

            image = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_GRAYSCALE)
            image = format_image(image)

            opticalFlowOverlay1(image_pv, image)


testOpenCVOpticalMoving()
