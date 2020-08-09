from __future__ import division

from app.other.Helper import *
import cv2
import app.Settings as Settings


def opticalFlowOverlay1(image_pv, image):
    image = cv2.convertScaleAbs(
        image, alpha=1.5, beta=60)

    # ---Approach 1---
    # ---Sharpening filter----
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    cv2.imshow('frame1', image)
    cv2.waitKey(0)


def testOpenCVOpticalMoving():
    def format_image(img):
        img = cv2.resize(
            img[250:-160, 100:-100], (0, 0), fx=2, fy=2)
        return img

    for _ in range(0, 20400, 10):

        for i in range(_, _ + 10):
            image_pv = cv2.imread(
                Settings.TRAIN_FRAMES + '/'
                + str(i) + '.jpg', cv2.IMREAD_COLOR)
            image_pv = format_image(image_pv)

            image = cv2.imread(
                Settings.TRAIN_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_COLOR)
            image = format_image(image)

            opticalFlowOverlay1(image_pv, image)


testOpenCVOpticalMoving()
