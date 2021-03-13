from __future__ import division

from app.Data import Data
from app.tools import Augmenters
from app.other.Helper import *
import cv2


def opticalFlowOverlay1(image_pv, image, image_next):
    one = calcOptical(image_pv, image)
    second = calcOptical(image, image_next)
    img_new = calcOptical(one, second)

    cv2.imshow('source', image_next)
    cv2.imshow('one', one)
    cv2.imshow('two', second)
    cv2.imshow('three', img_new)
    # cv2.imwrite('pyrLK-{}-source.png'.format(i), image)
    # cv2.imwrite('pyrLK-{}.png'.format(i), img_new)
    cv2.waitKey(0)
    return img_new


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
    new = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return new


def calcOptical2(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """

    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros_like(image_current)
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]

    # Flow Parameters
    #     flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)

    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow


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

            image_n = cv2.imread(
                Settings.TEST_FRAMES + '/'
                + str(i + 2) + '.jpg', cv2.IMREAD_COLOR)
            image_n = format_image(image_n)

            img_new = opticalFlowOverlay1(image_pv, image, image_n)
            # cv2.imshow('source', image_n)
            # cv2.imshow('frame1', img_new)
            # cv2.imwrite('pyrLK-{}-source.png'.format(i), image)
            # cv2.imwrite('pyrLK-{}.png'.format(i), img_new)
            # cv2.waitKey(0)


testOpenCVOpticalMoving()
