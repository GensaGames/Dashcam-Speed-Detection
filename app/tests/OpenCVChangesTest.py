import cv2
import numpy as np

from app.core import Augmenters
from app.other.Helper import *
from numpy import loadtxt
import cv2
import math
import app.Settings as Settings
import matplotlib.pyplot as plt


def test_image_resize():
    start_idx = 6606

    for idx in range(2):
        i = cv2.imread(Settings.TEST_FRAMES + '/' + str(
            start_idx + idx) + '.jpg', cv2.IMREAD_COLOR)

        # scale
        frm = cv2.resize(
            i[100:-175, 120:-120], (0, 0), fx=1, fy=1)

        cv2.imwrite('car-normal-ex3-' + str(idx) + '.jpg', frm)


test_image_resize()


def test_image_resize_mean():
    image1 = cv2.imread(Settings.TEST_FRAMES + '/9800.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread(Settings.TEST_FRAMES + '/46.jpg', cv2.IMREAD_COLOR)

    # frm = image1.mean(axis=2)
    frm = image2[100:-160, 80:-80]
    # frm = cv2.resize(frm, (320, 160))

    # image1 = image1[220:350, 0:640]
    # image2 = image2[220:350, 0:640]

    list1 = np.array([1, 2, 3, 4, 5, 6])
    print(list1[2:-2])

    # cv2.imwrite('sift_keypoints2.jpg', Augmenters.get_new_training().augment_image(frm))
    cv2.imwrite('sift_keypoints3.jpg', frm)

