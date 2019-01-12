import cv2
import numpy as np

from app.other.Helper import *
from numpy import loadtxt
import cv2
import math
import app.Settings as Settings
import matplotlib.pyplot as plt


def test1():

    start_idx = 6606
    for idx in range(5):
        i = cv2.imread('../../' + Settings.TEST_FRAMES + '/' + str(
            start_idx + idx) + '.jpg', cv2.IMREAD_COLOR)

        # scale
        frm = cv2.resize(
            i[190: -190, 220:-220], (0, 0), fx=1.3, fy=1.3)

        cv2.imwrite('car-normal-ex3-' + str(idx) + '.jpg', frm)


def test2():
    image1 = cv2.imread('../' + Settings.TEST_FRAMES + '/540.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread('../' + Settings.TEST_FRAMES + '/46.jpg', cv2.IMREAD_COLOR)

    frm = image1.mean(axis=2)
    frm = frm[230:-130, 220:-220]
    frm = cv2.resize(frm, (320, 160))

    # image1 = image1[220:350, 0:640]
    # image2 = image2[220:350, 0:640]

    list1 = np.array([1, 2, 3, 4, 5, 6])
    print(list1[2:-2])

    cv2.imwrite('sift_keypoints2.jpg', frm)
    # cv2.imwrite('sift_keypoints1.jpg', image2)


def test3():
    items = loadtxt(
        '../' + Settings.TRAIN_Y, delimiter=" ",
        unpack=False)

    items = np.reshape(
        items, (len(items), 1))

    changes = []
    for step in range(400, 20400, 100):

        direction = 0
        changes_idx = 1
        for i in range(step - 400 + 1, step):
            delta = items[i] - items[i - 1]
            changes_idx += 1

            if direction == 0 or math.copysign(
                    1, delta) == math.copysign(1, direction):
                direction = math.copysign(1, delta)
            else:
                changes.append(changes_idx)
                break

    fig, ax = plt.subplots()
    ax.plot(range (400, 20400, 100), changes)

    ax.set(xlabel='Frame Index Over Time',
           ylabel='Minimum Index on Momentum changes')

    annot_max(range (400, 20400, 100), np.array(changes), ax=ax)
    annot_min(range (400, 20400, 100), np.array(changes), ax=ax)
    annot_avr(np.array(changes), ax=ax)

    ax.grid()
    plt.show()

    plt.savefig('plot.png')


def test4():
    items = loadtxt(
        '../' + Settings.TRAIN_Y, delimiter=" ",
        unpack=False)

    items = np.reshape(
        items, (len(items), 1))

    change_items = []
    for idx, val in enumerate(items):
        if idx < 20:
            continue

        change_items.append(abs(val - items[idx - 20]))

    fig, ax = plt.subplots()
    ax.plot(
        range(20, 20400), change_items)

    ax.set(xlabel='Frame Index Over Time',
           ylabel='Delta Speed Changes from the last second (-20)')

    annot_max(range (20, 20400), np.array(change_items), ax=ax)
    annot_min(range (20, 20400), np.array(change_items), ax=ax)
    annot_avr(np.array(change_items), ax=ax)

    ax.grid()
    plt.show()

    plt.savefig('plot.png')


def test5():
    image = cv2.imread(
        '../../' + Settings.TRAIN_FRAMES + '/540.jpg', cv2.IMREAD_GRAYSCALE)

    from imgaug.augmenters import Sequential
    from imgaug.augmenters import Fliplr
    from imgaug.augmenters import GaussianBlur
    from imgaug.augmenters import GammaContrast
    from imgaug.augmenters import Invert
    from imgaug.augmenters import CoarseSalt
    from imgaug.augmenters import CoarseDropout
    from imgaug.augmenters import Emboss

    seq = Sequential([
        Fliplr(1.0), # to do
        GammaContrast(0.4), # from tow
        Emboss(0, strength=0.5), # from tow
        #Invert(1.0), # to do
        CoarseDropout(0.05, size_percent=0.02), # from to
        # CoarseSalt(0.05, size_percent=0.3), # from to
    ])
    seq1 = Sequential([
        Fliplr(1.0), # to do
        GammaContrast(1.0), # from tow
        Emboss(1, strength=1,), # from tow
        #Invert(1.0), # to do
        CoarseDropout(0.2, size_percent=0.02), # from to
        # CoarseSalt(0.05, size_percent=0.3), # from to
    ])

    a_t = seq.to_deterministic()
    image_n = a_t.augment_image(image)

    a_t1 = seq1.to_deterministic()
    image_n1 = a_t1.augment_image(image)

    cv2.imshow('image_n.jpg', image_n)
    cv2.imshow('image.jpg', image_n1)
    cv2.waitKey(0)


test5()


##################################################################
# Showing Timeline, where one source Frame and one
# Augmented, using current model from Project.
def test6():

    aug_model = AugmenterModel()

    start_index = 3490
    for _ in range(0, 20400, 1000):

        state = aug_model.model.to_deterministic()
        for i in range(_, _ + 10):
            image = cv2.imread(
                '../../' + Settings.TRAIN_FRAMES + '/'
                + str(start_index + i) + '.jpg', cv2.IMREAD_GRAYSCALE)

            cv2.imshow('Augmented', state.augment_image(image))
            cv2.imshow('Original', image)
            cv2.waitKey(0)


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


def test8():
    img = cv2.imread(
        '../../' + Settings.TRAIN_FRAMES + '/'
        + str(3700) + '.jpg', cv2.IMREAD_COLOR)
    rows,cols,ch = img.shape

    pts1 = np.float32([[50,50],[200,50],[100,200]])
    pts2 = np.float32([[50,50],[400,50],[0,200]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()





