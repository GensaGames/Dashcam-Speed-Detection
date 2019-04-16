from __future__ import division

import os

import math
import functools
import numpy as np
from numpy import loadtxt

import scipy as sp
import scipy.ndimage

from app import Settings
from app.other.LoggerFactory import get_logger
logger = get_logger()


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

        name1 = Settings.BUILD + '/' + 'post-v1.txt'
        map_step(source, name1,
                 Postprocessor.__fix_negative)

        name2 = Settings.BUILD + '/' + 'post-v2.txt'
        map_step(name1, name2,
                 functools.partial(
                     Postprocessor.__smooth_aggressive,
                     window=20, thr_increase=4, thr_decrease=8))

        name3 = Settings.BUILD + '/' + 'post-v3.txt'
        map_step(name2, name3,
                 functools.partial(
                     Postprocessor.__smooth, window=8))

        name4 = Settings.BUILD + '/' + 'post-v4.txt'
        map_step(name3, name4,
                 functools.partial(
                     Postprocessor.__smooth, window=6))

    @staticmethod
    def show_quality_deviation(source):

        values = []
        with open(source) as file:
            items = list(map(str.rstrip, file.readlines()))
            items = list(map(float, items))

            step_len = 15
            for i in range(0, len(items), step_len):
                val = np.std(items[i:i + step_len])
                values.append(val)

            logger.info('Mean of {} produced STD: {}'
                        .format(os.path.basename(source),
                                np.mean(values)))
        return values

    @staticmethod
    def __fix_negative(x):
        for idx, val in enumerate(x):
            if val < 0:
                x[idx] = 0.5
        return x

    @staticmethod
    def __smooth_aggressive(x, window, thr_increase, thr_decrease):
        previous_idx = 0
        near_idx_dict = {}
        start_near_idx = 0

        for idx, val in enumerate(x):
            if idx < window:
                continue
            avr = np.mean(
                x[idx - window:idx])

            should_stat = False
            changes = val - avr
            if changes > 0:
                if abs(changes) > thr_increase:
                    x[idx] = avr + (changes / 10)
                    should_stat = True
            else:
                if abs(changes) > thr_decrease:
                    x[idx] = avr + (changes / 10)
                    should_stat = True

            if should_stat:
                if start_near_idx == 0:
                    start_near_idx = idx
                if abs(idx - previous_idx) <= 2:
                    near_idx_dict[start_near_idx] = \
                        near_idx_dict.get(start_near_idx, 0) + 1
                else:
                    start_near_idx = 0
                previous_idx = idx

        logger.info('Changes Total len: {} Stats:\n{}'
                    .format(sum(near_idx_dict.values()), near_idx_dict))
        return x

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
    postprocessor.create_new(
        Settings.BUILD + '/' + 'OPT-V252-OPT-3D-CNN.txt')

    # postprocessor.show_quality_deviation(
    #     Settings.BUILD + '/' + 'OPT-V231-OPT-3D-CNN.txt')
    # postprocessor.show_quality_deviation(
    #     Settings.BUILD + '/' + 'OPT-V252-OPT-3D-CNN.txt')
