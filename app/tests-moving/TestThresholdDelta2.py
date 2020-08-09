from __future__ import division

from app.tools import Augmenters
from app.other.Helper import *
import cv2
import app.Settings as Settings


def opticalFlowOverlay1(image_pv, image):

    image_pv = cv2.adaptiveThreshold(
        image_pv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    new = diffImage(image_pv, image)
    cv2.imshow('frame1', new)
    cv2.waitKey(0)


def diffImage(image_pv, image):
    new = np.subtract(
        image.astype(np.int16),
        image_pv.astype(np.int16),
    )
    new = np.absolute(new)
    new = new.astype(np.uint8)
    return new


def testOpenCVOpticalMoving():
    def format_image(img, augmenter):
        img = augmenter.augment_image(img)

        img = cv2.resize(
            img[250:-160, 150:-150], (0, 0), fx=2, fy=2)
        return img

    for _ in range(1500, 20400, 10):

        for i in range(_, _ + 10):
            image_pv = cv2.imread(
                Settings.TRAIN_FRAMES + '/'
                + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
            img_aug = Augmenters.get_new_validation().image.to_deterministic()

            image_pv = format_image(image_pv, img_aug)

            image = cv2.imread(
                Settings.TRAIN_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_GRAYSCALE)
            image = format_image(image, img_aug)

            opticalFlowOverlay1(image_pv, image)


testOpenCVOpticalMoving()
