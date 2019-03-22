from __future__ import division

import math
import functools
import numpy as np
from numpy import loadtxt

import scipy as sp
import scipy.ndimage

from app import Settings
from app.other.LoggerFactory import get_logger


class Postprocessor:

    @staticmethod
    def create_new(source):

        def map_step(path_from, path_to, func):
            values = loadtxt(path_from, delimiter=" ",
                unpack=False)
            values = func(values)

            with open(path_to, "wb") as file:
                np.savetxt(
                    file, np.round(values, 8), fmt='%.8f',
                    delimiter="\n")

        # TODO(Postprocessing): Move to RX Actions

        name1 = Settings.BUILD + '/' + Postprocessor.\
            __fix_negative.__name__ + '1.txt'
        map_step(source, name1,
                 Postprocessor.__fix_negative)

        name2 = Settings.BUILD + '/' + Postprocessor. \
            __change_known_issue.__name__ + '2.txt'
        print('Local N1: ' + str(name1) + ' N2: ' + str(name2))
        map_step(name1, name2,
                 Postprocessor.__change_known_issue)

        # name3 = Settings.BUILD + '/' + Postprocessor. \
        #     __smooth_aggressive.__name__ + '3.txt'
        # map_step(name2, name3,
        #          functools.partial(
        #              Postprocessor.__smooth_aggressive,
        #              window=10, threshold=4))
        #
        # name4 = Settings.BUILD + '/' + Postprocessor. \
        #     __smooth.__name__ + '4.txt'
        # map_step(name3, name4,
        #          functools.partial(
        #              Postprocessor.__smooth, window=10))

    @staticmethod
    def show_quality_deviation(source):
        logger = get_logger()

        values = []
        with open(source) as file:
            items = list(map(str.rstrip, file.readlines()))
            items = list(map(float, items))

            step_len = 15
            for i in range(0, len(items), step_len):
                val = np.std(items[i:i + step_len])
                values.append(val)

            logger.info('Mean produced STD: {}'
                        .format(np.mean(values)))
        return values


    @staticmethod
    def __fix_negative(x):
        for idx, val in enumerate(x):
            if val < 0:
                x[idx] = 1.5
        return x

    @staticmethod
    def __smooth_aggressive(x, window, threshold):
        for idx, val in enumerate(x):
            if idx < window:
                continue
            avr = np.mean(
                x[idx - window:idx])

            changes = val - avr
            if abs(changes) > threshold:
                x[idx] = avr + (changes / 1.8)
        return x

    @staticmethod
    def __change_known_issue(values):
        indexes = np.concatenate(
            (np.arange(1080, 1720),
             np.arange(9640, 9840)))

        to_change = dict(zip(
            indexes, np.zeros(len(indexes))))

        for idx, val in enumerate(values):
            new_val = to_change.get(idx)
            if new_val is not None:
                values[idx] = new_val
        return values

    @staticmethod
    def __smooth(x, window):
        new_x = []
        for idx, val in enumerate(x):
            assert window % 2 == 0
            side = int(window / 2)

            if idx < side or idx + side + 1 > len(x):
                new_x.append(val)
                continue

            mean = np.mean(x[idx - side: idx + side + 1])
            new_x.append(mean)
        return new_x

    @staticmethod
    def __smooth_gaussian(x):
        return sp.ndimage.filters.gaussian_filter1d(
            x, 2, mode='constant')


#####################################
if __name__ == "__main__":
    postprocessor = Postprocessor()
    # postprocessor.create_new(
    #     Settings.BUILD + '/' + 'optical-3d-v121.txt')
    postprocessor.show_quality_deviation(
        Settings.BUILD + '/' + 'optical-3d-v120-n.txt')
    postprocessor.show_quality_deviation(
        Settings.BUILD + '/' + 'optical-3d-v121-n.txt')
