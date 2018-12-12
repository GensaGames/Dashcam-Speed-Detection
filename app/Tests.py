import cv2
import numpy as np
from app.Settings import TRAIN_FRAMES, TEST_FRAMES


def test1():

    start_idx = 380
    for idx in range(5):
        i = cv2.imread('../' + TEST_FRAMES + '/' + str(
            start_idx + idx) + '.jpg', cv2.IMREAD_COLOR)

        # scale
        frm = cv2.resize(
            i[250:-120, 220:-220], (0, 0), fx=1.5, fy=1.5)

        cv2.imwrite('scale-' + str(idx) + '.jpg', frm)


test1()


def test2():
    image1 = cv2.imread('../' + TEST_FRAMES + '/88.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread('../' + TEST_FRAMES + '/46.jpg', cv2.IMREAD_COLOR)

    frm = image1.mean(axis=2)
    frm = frm[190:-190, 220:-220]
    frm = cv2.resize(frm, (320, 160))

    # image1 = image1[220:350, 0:640]
    # image2 = image2[220:350, 0:640]

    list1 = np.array([1, 2, 3, 4, 5, 6])
    print(list1[2:-2])

    cv2.imwrite('sift_keypoints2.jpg', frm)
    # cv2.imwrite('sift_keypoints1.jpg', image2)





