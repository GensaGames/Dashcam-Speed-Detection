import cv2
import numpy as np

from app.core import Augmenters
from app.other.Helper import *
from numpy import loadtxt
import cv2
import math
import app.Settings as Settings
import matplotlib.pyplot as plt


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


##################################################################
# Showing Timeline, where one source Frame and one
# Augmented, using current model from Project.
def test6():

    aug_model = Augmenters.get_new_validation()

    start_index = 3490
    for _ in range(0, 20400, 1000):

        state = aug_model.to_deterministic()
        for i in range(_, _ + 10):
            image = cv2.imread(
                '../../' + Settings.TRAIN_FRAMES + '/'
                + str(start_index + i) + '.jpg', cv2.IMREAD_GRAYSCALE)

            cv2.imshow('Augmented', state.augment_image(image))
            cv2.imshow('Original', image)
            cv2.waitKey(0)
test6()

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





