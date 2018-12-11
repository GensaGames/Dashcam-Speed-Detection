import cv2
import numpy as np
from app.Settings import TRAIN_FRAMES


def test1():
    image1 = cv2.imread('../' + TRAIN_FRAMES + '/20219.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread('../' + TRAIN_FRAMES + '/20220.jpg', cv2.IMREAD_COLOR)
    image3 = cv2.imread('../' + TRAIN_FRAMES + '/20221.jpg', cv2.IMREAD_COLOR)
    image4 = cv2.imread('../' + TRAIN_FRAMES + '/20222.jpg', cv2.IMREAD_COLOR)
    image5 = cv2.imread('../' + TRAIN_FRAMES + '/20223.jpg', cv2.IMREAD_COLOR)

    list1 = [image1, image2, image3, image4, image5]
    for idx, i in enumerate(list1):

        # source
        cv2.imwrite('scale/scale-source-' + str(idx) + '.jpg', i[130:-130, 160:-160])

        # scale
        frm = cv2.resize(
            i[130:-130, 160:-160], (0, 0), fx=0.5, fy=0.5)

        cv2.imwrite('scale/scale-' + str(idx) + '.jpg', frm)

    cv2.imwrite('sift_keypoints1.jpg', image2)


def test2():
    image1 = cv2.imread('../' + TRAIN_FRAMES + '/207.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread('../' + TRAIN_FRAMES + '/208.jpg', cv2.IMREAD_COLOR)

    frm = image1.mean(axis=2)
    frm = frm[190:-190, 220:-220]
    frm = cv2.resize(frm, (320, 160))

    # image1 = image1[220:350, 0:640]
    # image2 = image2[220:350, 0:640]

    list1 = np.array([1, 2, 3, 4, 5, 6])
    print(list1[2:-2])

    cv2.imwrite('sift_keypoints.jpg', frm)
    # cv2.imwrite('sift_keypoints1.jpg', image2)



