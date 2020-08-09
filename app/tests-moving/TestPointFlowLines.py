from __future__ import division

from app.tools import Augmenters
from app.other.Helper import *
import cv2
import app.Settings as Settings


def opticalFlowOverlay2(imagePv, image):
    feature_params = dict(
        maxCorners=500,
        qualityLevel=0.1,
        minDistance=7,
        blockSize=5
    )
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS
                  | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    imagePvGray = cv2.cvtColor(imagePv, cv2.COLOR_RGB2GRAY)
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Finds edges in an image using the [Canny86] algorithm.
    p0 = cv2.goodFeaturesToTrack(imageGray, mask=None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        imagePvGray, imageGray, p0, None, **lk_params)
    color = np.random.randint(0, 255, (300, 3))

    colorL = np.array([122, 122, 122])
    colorC = np.array([255, 255, 255])

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    mask = np.zeros_like(image)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        image = cv2.line(mask, (a, b), (c, d), colorL.tolist(), 1)

    return cv2.add(image, mask)


def testOpenCVOpticalMoving():
    def format_image(img):

        ia = Augmenters.get_new_validation().image
        img = ia.augment_image(img)

        img = cv2.resize(
            img[80:-160, 60:-60], (0, 0), fx=1.5, fy=1.5)
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
