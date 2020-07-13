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
    gray_pv = cv2.cvtColor(image_pv, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    cornersPv = cv2.goodFeaturesToTrack(gray_pv, 250, 0.01, 10)
    cornersPv = np.int0(cornersPv)

    corners = cv2.goodFeaturesToTrack(gray, 250, 0.01, 10)
    corners = np.int0(corners)

    for i in cornersPv:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 3, [122, 122, 122], -1)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 3, [255, 255, 255], -1)
    return image


def opticalFlowOverlay2(image_pv, image):
    gray_pv = cv2.cvtColor(image_pv, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.zeros_like(image)

    cornersPv = cv2.goodFeaturesToTrack(gray_pv, 250, 0.01, 10)
    cornersPv = np.int0(cornersPv)

    corners = cv2.goodFeaturesToTrack(gray, 250, 0.01, 10)
    corners = np.int0(corners)

    for i in cornersPv:
        x, y = i.ravel()
        cv2.circle(mask, (x, y), 3, [122, 122, 122], -1)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(mask, (x, y), 3, [255, 255, 255], -1)
    return mask


def testOpenCVOpticalMoving():

    def format_image(img):
        ia = Augmenters.get_new_validation().image
        img = ia.augment_image(img)

        img = cv2.resize(
            img[80:-160, 60:-60], (0, 0), fx=1, fy=1)
        return img

    for _ in range(100, 20400, 20):

        for i in range(_, _ + 10):
            image_pv = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i) + '.jpg', cv2.IMREAD_COLOR)
            image_pv = format_image(image_pv)

            image = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_COLOR)
            image = format_image(image)

            img_new = opticalFlowOverlay2(image_pv, image)
            cv2.imshow('frame1', img_new)
            cv2.waitKey(0)


testOpenCVOpticalMoving()
