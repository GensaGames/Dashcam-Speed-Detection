import logging
import time

import cv2

logging.basicConfig(level=logging.INFO)

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
            Fliplr(p=0.5),
            Invert(p=0.2),
            SomeOf((0, 1), [
                CoarsePepper(
                    p=(0.05, 0.2), size_percent=(0.1, 0.5)),
                CoarseSalt(
                    p=(0.05, 0.2), size_percent=(0.1, 0.5)),
            ]),
        ])

    @property
    def model(self):
        return self._aug_model


#####################################
if __name__ == "__main__":
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
