
import time

from imgaug.augmenters import CoarseDropout
from imgaug.augmenters import GammaContrast
from imgaug.augmenters import Sequential
from imgaug.augmenters import Sometimes
from imgaug.augmenters import ia


def get_new_training():
    ia.seed(int(time.time()))

    return Sequential([
        GammaContrast(gamma=(0.35, 0.9)),
        Sometimes(0.3, CoarseDropout(
            p=(0.05, 0.3), size_percent=(0.1, 0.5))),
    ])


def get_new_validation():
    return Sequential([
        GammaContrast(gamma=0.45),
    ])


#####################################
if __name__ == "__main__":
    assert get_new_training() is not None
