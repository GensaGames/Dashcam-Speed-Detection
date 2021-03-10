from __future__ import division

import math

from app.tools import Augmenters
from app.other.Helper import *
import cv2
import app.Settings as Settings


def opticalFlowOverlay2(image_pv, image):
    feature_params = dict(
        maxCorners=0,
        qualityLevel=0.1,
        minDistance=5,
        blockSize=5
    )
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS
                  | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    gray_pv = cv2.cvtColor(image_pv, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Finds edges in an image using the [Canny86] algorithm.
    p0 = cv2.goodFeaturesToTrack(gray_pv, mask=None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        gray_pv, gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    def filter_points(v1, v2):
        x_change = v2[0] - v1[0]
        y_change = v2[1] - v1[1]
        return y_change > 0 and abs(x_change) < 0.5

    norms = []
    # Find average Vector Norm
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        norms.append(math.hypot(c - a, d - b))
    norm_mean = np.mean(norms)

    mask = np.zeros_like(image)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        norm = math.hypot(c - a, d - b)

        color = [33, 33, 255] if filter_points((a, b), (c, d)) \
            else [255, 255, 255]

        color = [255, 33, 33] if norm > norm_mean * 7 \
            else color

        mask = cv2.line(mask, (a, b), (c, d), color, 1)
        mask = cv2.circle(mask, (c, d), 2, color, -1)

    return mask


def testOpenCVOpticalMoving():
    def format_image(img):

        ia = Augmenters.get_new_validation()
        img = ia.augment_image(img)

        img = cv2.resize(
            img[160:-160, 100:-100], (0, 0), fx=1.5, fy=1.5)
        return img

    for _ in range(1700, 20400, 20):

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
            cv2.imshow('source', image)
            cv2.imshow('frame1', img_new)
            # cv2.imwrite('pyrLK-{}-source.png'.format(i), image)
            # cv2.imwrite('pyrLK-{}.png'.format(i), img_new)
            cv2.waitKey(0)


testOpenCVOpticalMoving()
