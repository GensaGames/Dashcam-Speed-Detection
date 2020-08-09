from __future__ import division

from app.tools import Augmenters
from app.other.Helper import *
import cv2
import app.Settings as Settings


def opticalFlowOverlay1(image_pv, image):
    optical = calcOptical(image_pv, image)

    cv2.imshow('frame1', optical)
    cv2.waitKey(0)


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
    augmenter = Augmenters.get_new_training()

    def format_image(img):
        img = cv2.resize(
            img[230:-150, 180:-180], (0, 0), fx=1.5, fy=1.5)
        return img

    for _ in range(10, 20400, 10):

        for i in range(_, _ + 10):
            img_aug = augmenter.image.to_deterministic()

            image_pv = cv2.imread(
                Settings.TRAIN_FRAMES + '/'
                + str(i) + '.jpg', cv2.IMREAD_COLOR)
            image_pv = format_image(
                img_aug.augment_image(image_pv)
            )

            image = cv2.imread(
                Settings.TRAIN_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_COLOR)
            # cv2.imshow('source', format_image(image))

            image = format_image(
                img_aug.augment_image(image)
            )

            opticalFlowOverlay1(image_pv, image)


testOpenCVOpticalMoving()
