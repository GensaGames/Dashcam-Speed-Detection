import logging
import time

import cv2


from imgaug.augmenters import Sequential
from imgaug.augmenters import ia
from imgaug.augmenters import Sometimes
from imgaug.augmenters import GammaContrast
from imgaug.augmenters import Emboss
from imgaug.augmenters import CoarseSalt
from imgaug.augmenters import CoarseDropout
from imgaug.augmenters import SomeOf

from app import Settings


def get_new_training():
    ia.seed(int(time.time()))

    return Sequential([
        GammaContrast(gamma=(0.2, 1)),
        Sometimes(0.3, CoarseDropout(
            p=(0.05, 0.2), size_percent=(0.1, 0.3))),
    ])


def get_new_validation():
    return Sequential([
        GammaContrast(gamma=0.4),
    ])


#####################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    assert get_new_training() is not None
