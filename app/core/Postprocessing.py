from __future__ import division

import math

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

    @staticmethod
    def smooth(x):
        return sp.ndimage.filters.gaussian_filter1d(
            x, 2, mode='constant')


#####################################
if __name__ == "__main__":

    def test_smooth_aggressive():
        values = loadtxt(
            '../../' + Settings.BUILD + '/' +
            'v1-test.txt', delimiter=" ",
            unpack=False)
        Postprocessor.smooth_aggressive(values, 5, 1)

        path_to = '../../' + Settings.BUILD + '/' + \
                  'v1-test-new-1.txt'
        with open(path_to, "wb") as file:
            np.savetxt(
                file, np.round(values, 8), fmt='%.8f',
                delimiter="\n")

    test_smooth_aggressive()

    def test_change_known_issue():
        values = loadtxt(
            '../../' + Settings.BUILD + '/' +
            'v1-test-new-1.txt', delimiter=" ",
            unpack=False)
        Postprocessor.change_known_issue(values)

        path_to = '../../' + Settings.BUILD + '/' + \
                  'v1-test-new-2.txt'
        with open(path_to, "wb") as file:
            np.savetxt(
                file, np.round(values, 8), fmt='%.8f',
                delimiter="\n")

    test_change_known_issue()

    def test_smooth():
        values = loadtxt(
            '../../' + Settings.BUILD + '/' +
            'v1-test-new-2.txt', delimiter=" ",
            unpack=False)
        values = Postprocessor.smooth(values)

        path_to = '../../' + Settings.BUILD + '/' + \
                  'v1-test-new-3.txt'
        with open(path_to, "wb") as file:
            np.savetxt(
                file, np.round(values, 8), fmt='%.8f',
                delimiter="\n")

    test_smooth()

