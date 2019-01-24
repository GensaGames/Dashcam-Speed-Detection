from __future__ import division

import math
import functools
import numpy as np
from numpy import loadtxt

import scipy as sp
import scipy.ndimage

from app import Settings


class Postprocessor:

    @staticmethod
    def smooth_aggressive(x, window, threshold):
        for idx, val in enumerate(x):
            if idx < window:
                continue
            avr = np.mean(
                x[idx-window:idx])

            if abs(val - avr) > threshold:
                sign = math.copysign(1, val - avr)
                x[idx] = avr + (sign * threshold)
        return x

    @staticmethod
    def change_known_issue(x):
        indexes = np.concatenate(
            (np.arange(1080, 1720),
             np.arange(9640, 9840)))

        to_change = dict(zip(
            indexes, np.zeros(len(indexes))))

        for idx, val in enumerate(x):
            new_val = to_change.get(idx)
            if new_val is not None:
                x[idx] = new_val
        return x

    @staticmethod
    def map_to_mean(x, window):
        new_x = []
        for idx, val in enumerate(x):
            assert window % 2 != 0
            side = int(window / 2)

            if idx < side or idx + side + 1 > len(x):
                new_x.append(val)
                continue

            mean = np.mean(x[idx - side: idx + side + 1])
            new_x.append(mean)
        return new_x

    @staticmethod
    def smooth_gaussian(x):
        return sp.ndimage.filters.gaussian_filter1d(
            x, 2, mode='constant')


#####################################
if __name__ == "__main__":

    def test_map(path_from, path_to, func):
        values = loadtxt(
            '../../' + Settings.BUILD + '/' +
            path_from, delimiter=" ",
            unpack=False)
        values = func(values)

        path_to = '../../' + Settings.BUILD + '/' + path_to
        with open(path_to, "wb") as file:
            np.savetxt(
                file, np.round(values, 8), fmt='%.8f',
                delimiter="\n")

    test_map(
        'v1-test.txt', 'v1-test-new-1.txt',
        functools.partial(
            Postprocessor.smooth_aggressive, window=5, threshold=1))

    test_map(
        'v1-test-new-1.txt', 'v1-test-new-2.txt',
        Postprocessor.change_known_issue)

    test_map(
        'v1-test-new-2.txt', 'v1-test-new-3.txt',
        functools.partial(
            Postprocessor.map_to_mean, window=5))

    test_map(
        'v1-test-new-3.txt', 'v1-test-new-4.txt',
        Postprocessor.smooth_gaussian)



