
import time

from imgaug.augmenters import CoarseDropout
from imgaug.augmenters import GammaContrast
from imgaug.augmenters import Sequential
from imgaug.augmenters import Sometimes
from imgaug.augmenters import ia


def get_new_training():

    return Sequential([
        GammaContrast(gamma=(0.45, 0.7)),
    ])


def get_new_validation():
    return Sequential([
        GammaContrast(gamma=(0.45, 0.7)),
    ])


#####################################
if __name__ == "__main__":
    assert get_new_training() is not None
