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
        image_pv, alpha=1.5, beta=60)
    image = cv2.convertScaleAbs(
        image, alpha=1.5, beta=60)

    new = np.subtract(
        image,
        image_pv
    )
    cv2.imshow('frame1', new)
    cv2.waitKey(0)


def testOpenCVOpticalMoving():
    def format_image(img, augmenter):
        img = augmenter.augment_image(img)

        img = cv2.resize(
            img[250:-160, 150:-150], (0, 0), fx=2, fy=2)
        return img

    for _ in range(2000, 20400, 10):

        for i in range(_, _ + 10):
            image_pv = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
            img_aug = Augmenters.get_new_validation().image.to_deterministic()

            image_pv = format_image(image_pv, img_aug)

            image = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_GRAYSCALE)
            image = format_image(image, img_aug)

            opticalFlowOverlay1(image_pv, image)


testOpenCVOpticalMoving()
