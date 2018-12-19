import imgaug as ia
from imgaug import augmenters as iaa
import random
import numpy as np
import cv2
import app.Settings as Settings


def test1():
    start_idx = 580
    images = []
    for idx in range(5):
        i = cv2.imread('../../' + Settings.TEST_FRAMES + '/' + str(
            start_idx + idx) + '.jpg', cv2.IMREAD_COLOR)

        frm = cv2.resize(
            i[190: -190, 220:-220], (0, 0), fx=1.3, fy=1.3)
        images.append(frm)

    images = np.array(images)

    # Generate random keypoints.
    # The augmenters expect a list of imgaug.KeypointsOnImage.
    keypoints_on_images = []
    for image in images:
        height, width = image.shape[0:2]
        keypoints = []
        for _ in range(4):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            keypoints.append(ia.Keypoint(x=x, y=y))
        keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))

    seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(scale=(0.5, 0.7))])
    seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start

    # augment keypoints and images
    images_aug = seq_det.augment_images(images)
    keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)

    # Example code to show each image and print the new keypoints coordinates
    for img_idx, (image_before, image_after, keypoints_before, keypoints_after) in enumerate(zip(images, images_aug, keypoints_on_images, keypoints_aug)):
        image_before = keypoints_before.draw_on_image(image_before)
        image_after = keypoints_after.draw_on_image(image_after)
        ia.imshow(np.concatenate((image_before, image_after), axis=1)) # before and after
        for kp_idx, keypoint in enumerate(keypoints_after.keypoints):
            keypoint_old = keypoints_on_images[img_idx].keypoints[kp_idx]
            x_old, y_old = keypoint_old.x, keypoint_old.y
            x_new, y_new = keypoint.x, keypoint.y
            print("[Keypoints for image #%d] before aug: x=%d y=%d | after aug: x=%d y=%d" % (img_idx, x_old, y_old, x_new, y_new))

test1()

def test2():

    start_idx = 580
    images = []
    for idx in range(5):
        i = cv2.imread('../../' + Settings.TEST_FRAMES + '/' + str(
            start_idx + idx) + '.jpg', cv2.IMREAD_COLOR)

        frm = cv2.resize(
            i[190: -190, 220:-220], (0, 0), fx=1.3, fy=1.3)
        images.append(frm)

    images = np.array(images)
