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
def test7():
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