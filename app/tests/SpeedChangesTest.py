import cv2
import numpy as np

from app.other.Helper import *
from numpy import loadtxt
import cv2
import math
import app.Settings as Settings
import matplotlib.pyplot as plt


def test_momentum_changes_over_time():
    items = loadtxt(
        '../' + Settings.TRAIN_Y, delimiter=" ",
        unpack=False)

    items = np.reshape(
        items, (len(items), 1))

    changes = []
    for step in range(400, 20400, 100):

        direction = 0
        changes_idx = 1
        for i in range(step - 400 + 1, step):
            delta = items[i] - items[i - 1]
            changes_idx += 1

            if direction == 0 or math.copysign(
                    1, delta) == math.copysign(1, direction):
                direction = math.copysign(1, delta)
            else:
                changes.append(changes_idx)
                break

    fig, ax = plt.subplots()
    ax.plot(range (400, 20400, 100), changes)

    ax.set(xlabel='Frame Index Over Time',
           ylabel='Minimum Index on Momentum changes')

    annot_max(range (400, 20400, 100), np.array(changes), ax=ax)
    annot_min(range (400, 20400, 100), np.array(changes), ax=ax)
    annot_avr(np.array(changes), ax=ax)

    ax.grid()
    plt.show()

    plt.savefig('plot.png')


def test_speed_changes_over_time():
    items = loadtxt(
        '../../' + Settings.BUILD + '/'
        + 'v1-test-new-5.txt', delimiter=" ",
        unpack=False)

    items = np.reshape(
        items, (len(items), 1))

    change_items = []
    for idx, val in enumerate(items):
        if idx < 1:
            continue

        change_items.append(abs(val - items[idx - 1]))

    fig, ax = plt.subplots()
    ax.plot(
        range(1, 10798), change_items)

    ax.set(xlabel='Frame Index Over Time',
           ylabel='Delta Speed Changes from the last second (-20)')

    annot_max(range (1, 10798), np.array(change_items), ax=ax)
    annot_min(range (1, 10798), np.array(change_items), ax=ax)
    annot_avr(np.array(change_items), ax=ax)

    ax.grid()
    plt.show()

    plt.savefig('plot1.png')


test_speed_changes_over_time()


def test_local_smooth():
    list1 = np.array([4, 4, 4, 4, -19, -18, -18, -17, -16])

    def smooth(x, window, threshold):
        for idx, val in enumerate(x):
            if idx < window:
                continue
            avr = np.mean(
                x[idx-window:idx])

            if abs(val - avr) > threshold:
                sign = math.copysign(
                    1, val - avr)
                x[idx] = avr + (sign * threshold)

    smooth(list1, 3, 2)
    print(list1)


