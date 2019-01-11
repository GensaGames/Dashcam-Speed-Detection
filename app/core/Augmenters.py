import logging
import time

import cv2


from imgaug.augmenters import Sequential
from imgaug.augmenters import ia
from imgaug.augmenters import Fliplr
from imgaug.augmenters import GammaContrast
from imgaug.augmenters import Invert
from imgaug.augmenters import CoarseSalt
from imgaug.augmenters import CoarsePepper
from imgaug.augmenters import SomeOf

from app import Settings


class AugmenterModel:

    def __init__(self):
        ia.seed(int(time.time()))
        self._aug_model = self.get_new_aug()

    @staticmethod
    def get_new_aug():
        return Sequential([
            GammaContrast(gamma=(0.2, 1.2)),
            SomeOf((0, 1), [
                CoarsePepper(
                    p=(0.05, 0.2), size_percent=(0.1, 0.3)),
                CoarseSalt(
                    p=(0.05, 0.2), size_percent=(0.1, 0.3)),
            ]),
        ])

    @property
    def model(self):
        return self._aug_model


#####################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    aug_model = AugmenterModel()
    assert aug_model.model is not None
