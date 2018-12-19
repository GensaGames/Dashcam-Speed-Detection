import cv2
import numpy as np
from app.other.Helper import *
from numpy import loadtxt
import cv2
import math
import app.Settings as Settings
import matplotlib.pyplot as plt


def test1():

    start_idx = 6606
    for idx in range(5):
        i = cv2.imread('../../' + Settings.TEST_FRAMES + '/' + str(
            start_idx + idx) + '.jpg', cv2.IMREAD_COLOR)

        # scale
        frm = cv2.resize(
            i[190: -190, 220:-220], (0, 0), fx=1.3, fy=1.3)

        cv2.imwrite('car-normal-ex3-' + str(idx) + '.jpg', frm)


test1()


def test2():
    image1 = cv2.imread('../' + Settings.TEST_FRAMES + '/540.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread('../' + Settings.TEST_FRAMES + '/46.jpg', cv2.IMREAD_COLOR)

    frm = image1.mean(axis=2)
    frm = frm[230:-130, 220:-220]
    frm = cv2.resize(frm, (320, 160))

    # image1 = image1[220:350, 0:640]
    # image2 = image2[220:350, 0:640]

    list1 = np.array([1, 2, 3, 4, 5, 6])
    print(list1[2:-2])

    cv2.imwrite('sift_keypoints2.jpg', frm)
    # cv2.imwrite('sift_keypoints1.jpg', image2)


def test3():
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



def test4():
    items = loadtxt(
        '../' + Settings.TRAIN_Y, delimiter=" ",
        unpack=False)

    items = np.reshape(
        items, (len(items), 1))

    change_items = []
    for idx, val in enumerate(items):
        if idx < 20:
            continue

        change_items.append(abs(val - items[idx - 20]))

    fig, ax = plt.subplots()
    ax.plot(
        range(20, 20400), change_items)

    ax.set(xlabel='Frame Index Over Time',
           ylabel='Delta Speed Changes from the last second (-20)')

    annot_max(range (20, 20400), np.array(change_items), ax=ax)
    annot_min(range (20, 20400), np.array(change_items), ax=ax)
    annot_avr(np.array(change_items), ax=ax)

    ax.grid()
    plt.show()

    plt.savefig('plot.png')






