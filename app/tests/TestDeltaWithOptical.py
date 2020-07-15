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

    delta = np.subtract(
        image.astype(np.int16),
        image_pv.astype(np.int16),
    )
    new = np.absolute(delta)
    new = new.astype(np.uint8)
    cv2.imshow('frame1', new)
    cv2.waitKey(0)


def diffImage(image_pv, image):
    # new = cv2.adaptiveThreshold(
    #     new, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY, 11, 2
    # )
    delta = np.subtract(
        image.astype(np.int16),
        image_pv.astype(np.int16),
    )
    new = np.absolute(delta)
    new = new.astype(np.uint8)
    return new


def calcOptical(img1, img2):
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
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return hsv


def testOpenCVOpticalMoving():
    def format_image(img):
        img = cv2.resize(
            img[250:-160, 150:-150], (0, 0), fx=2, fy=2)
        return img

    for _ in range(100, 20400, 10):

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
