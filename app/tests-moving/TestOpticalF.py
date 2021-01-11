from __future__ import division

from app.Data import Data
from app.tools import Augmenters
from app.other.Helper import *
import cv2


def opticalFlowOverlay1(image_pv, image):
    optical = calcOptical(image_pv, image)

    cv2.imshow('frame1', optical)
    cv2.imshow('frame2', image)

    img3 = cv2.resize(
        image, (220, 66), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame3', img3)
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
    augmenter = Augmenters.get_new_validation()

    def format_image(img):
        # img = cv2.resize(
        #     img[150:-150, 50:-50], (220, 66), interpolation=cv2.INTER_AREA)

        img = cv2.resize(
            img[150:-150, 50:-50], (0, 0), fx=1, fy=1)
        return img

    sources = list(filter(
        lambda x: x.name == 'Default',
        Data().get_sources()
    ))

    while True:
        s = sources[np.random.randint(0, len(sources))]
        print('Using Source: {}'.format(s.name))

        start = np.random.randint(10, len(s.y_values))
        for i in range(start, start + 30):
            img_aug = augmenter.image.to_deterministic()

            image_pv = cv2.imread(s.path + '/'
                + str(i) + '.jpg', cv2.IMREAD_COLOR)
            image_pv = format_image(
                img_aug.augment_image(image_pv)
            )

            image = cv2.imread(s.path + '/'
                + str(i + 1) + '.jpg', cv2.IMREAD_COLOR)
            image = format_image(
                img_aug.augment_image(image)
            )
            print('Frame Idx: {}. Speed: {}'.format(
                i, s.y_values[i]
            ))
            opticalFlowOverlay1(image_pv, image)


testOpenCVOpticalMoving()
