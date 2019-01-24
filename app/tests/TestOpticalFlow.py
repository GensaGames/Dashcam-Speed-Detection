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


##################################################################
# input: image_current, image_next (RGB images)
# calculates optical flow magnitude and angle and places it into HSV image
# * Set the saturation to the saturation value of image_next
# * Set the hue to the angles returned from computing the flow params
# * set the value to the magnitude returned from computing the flow params
# * Convert from HSV to RGB and return RGB image with same size as original image
def test_opencv_optical_source():
    image_current = cv2.imread(
        '../../' + Settings.TRAIN_FRAMES + '/'
        + str(3700) + '.jpg', cv2.IMREAD_COLOR)

    image_next = cv2.imread(
        '../../' + Settings.TRAIN_FRAMES + '/'
        + str(3701) + '.jpg', cv2.IMREAD_COLOR)
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)

    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros(image_current.shape)
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]

    # Flow Parameters
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

    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype= np.float32)

    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    cv2.imshow('Original', rgb_flow)
    cv2.waitKey(0)

    return rgb_flow



def test_opencv_optical1():
    frame1 = cv2.imread(
        '../../' + Settings.TRAIN_FRAMES + '/'
        + str(3700) + '.jpg', cv2.IMREAD_COLOR)

    image_current = cv2.imread(
        '../../' + Settings.TRAIN_FRAMES + '/'
        + str(3700) + '.jpg', cv2.IMREAD_GRAYSCALE)

    image_next = cv2.imread(
        '../../' + Settings.TRAIN_FRAMES + '/'
        + str(3701) + '.jpg', cv2.IMREAD_GRAYSCALE)

    flow = cv2.calcOpticalFlowFarneback(
         image_current,image_next, None,
        0.5, 3, 15, 3, 5, 1.2, 0)

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv = np.asarray(hsv, dtype= np.float32)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',bgr)
    cv2.waitKey(0)


def test_opencv_optical_moving():
    start_index = 2000
    for _ in range(0, 20400, 1000):

        for i in range(_, _ + 10):
            ia = Augmenters.get_new_validation()

            image_current = cv2.imread(
                '../../' + Settings.TEST_FRAMES + '/'
                + str(start_index + i) + '.jpg', cv2.IMREAD_COLOR)
            image_current = ia.augment_image(image_current)

            image_next = cv2.imread(
                '../../' + Settings.TEST_FRAMES + '/'
                + str(start_index + i + 1) + '.jpg', cv2.IMREAD_COLOR)
            image_next = ia.augment_image(image_next)

            hsv = np.zeros_like(image_current)
            # set saturation
            hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]

            # Flow Parameters
            flow_mat = None
            image_scale = 0.5
            nb_images = 3
            win_size = 15
            nb_iterations = 2
            deg_expansion = 5
            STD = 1.3

            # obtain dense optical flow paramters
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY),

                flow_mat, image_scale, nb_images, win_size, nb_iterations,
                deg_expansion, STD, 0)

            # convert from cartesian to polar
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # hue corresponds to direction
            hsv[:,:,0] = ang * (180/ np.pi / 2)

            # value corresponds to magnitude
            hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

            # convert HSV to float32's
            hsv = np.asarray(hsv, dtype= np.float32)

            hsv = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

            cv2.imshow('frame2',hsv)
            cv2.waitKey(0)


test_opencv_optical_moving()
