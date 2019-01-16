from __future__ import division

import math

import numpy as np
from numpy import loadtxt

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
    def change_known_issue(x, window, threshold):
        pass


#####################################
if __name__ == "__main__":
    values = loadtxt(
        '../../' + Settings.BUILD + '/' +
        Settings.BUILT_TEST, delimiter=" ",
        unpack=False)
    Postprocessor.smooth_aggressive(values, 20, 2)

    path_to = '../../' + Settings.BUILD + '/' +\
              Settings.BUILT_TEST_PR1
    with open(path_to, "wb") as file:
        np.savetxt(
            file, np.round(values, 8), fmt='%.8f',
            delimiter="\n")


