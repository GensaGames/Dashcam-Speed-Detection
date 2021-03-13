from __future__ import division

import cv2
import app.Settings as Settings


def opticalFlowOverlay1(image_pv, image):
    image = cv2.convertScaleAbs(
        image, alpha=1.5, beta=50)

    new = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return cv2.bitwise_not(new)


def testOpenCVOpticalMoving():
    def format_image(img):
        img = cv2.resize(
            img[160:-160, 100:-100], (0, 0), fx=1.5, fy=1.5)
        return img

    for _ in range(1700, 20400, 10):

        for i in range(_, _ + 10):
            image_pv = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
            image_pv = format_image(image_pv)

            image = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_GRAYSCALE)
            image = format_image(image)

            img_new = opticalFlowOverlay1(image_pv, image)
            cv2.imshow('source', image)
            cv2.imshow('frame1', img_new)
            # cv2.imwrite('pyrLK-{}-source.png'.format(i), image)
            cv2.imwrite('pyrLK-{}.png'.format(i), img_new)
            cv2.waitKey(0)


testOpenCVOpticalMoving()
