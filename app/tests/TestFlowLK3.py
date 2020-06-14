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


# noinspection DuplicatedCode
def calcOptical(img1, img2):
    hsv = np.zeros_like(img1)
    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(
        img2, cv2.COLOR_RGB2HSV)[:, :, 1]

    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None,
        0.6, 4, 20, 2, 5, 1.1, 0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(
        flow[..., 0], flow[..., 1])

    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)

    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(
        mag, None, 0, 255, cv2.NORM_MINMAX)

    # Сonvert HSV to float32's
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return hsv


# noinspection DuplicatedCode
def testOpenCVOpticalMoving():
    for i in range(67, 1000):

        ia = Augmenters.get_new_validation()

        image_pv = cv2.imread(
            Settings.TRAIN_FRAMES + '/'
            + str(i - 1) + '.jpg', cv2.IMREAD_COLOR)
        image_pv = ia.augment_image(image_pv)
        image_pv = cv2.resize(
            image_pv[150:-200, 200:-200], (0, 0), fx=1.5, fy=1.5)

        image_next = cv2.imread(
            Settings.TRAIN_FRAMES + '/'
            + str(i) + '.jpg', cv2.IMREAD_COLOR)
        image_next = ia.augment_image(image_next)
        image_next = cv2.resize(
            image_next
            [150:-200, 200:-200], (0, 0), fx=1.5, fy=1.5)

        img_new = calcOptical(image_pv, image_next)
        cv2.imshow('frame1', img_new)
        cv2.waitKey(0)


testOpenCVOpticalMoving()
