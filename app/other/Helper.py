import datetime
import os

import cv2
import jsonpickle
import numpy as np
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

import app.Settings as Settings
from keras.models import model_from_json
from keras.models import load_model


##########################################
def save_plot_with(path_to, plot, prefix, model, p_params):
    if not os.path.exists(path_to):
        os.makedirs(path_to)

    name = path_to + prefix + '-' + str(type(model).__name__) + '-LookBack-' \
           + str(p_params.backward) + '-Layers-' + str('~') \
           + '-LR-' + str('~') + '-Units-' + str('~')
    plot.savefig(name + '.png')


##########################################
def save_plot(path_to, plot, prefix):
    if not os.path.exists(path_to):
        os.makedirs(path_to)
    plot.savefig(path_to + prefix + '.png')


##########################################
def backup_model_with(path_to, name, model, *args):
    path_to = path_to + '/' + name + '/'
    if not os.path.exists(path_to):
        os.makedirs(path_to)

    model.save(path_to + Settings.BUILD_MODEL)
    for i in args:
        with open(path_to + type(i).__name__, "w+") as file:
            file.write(jsonpickle.encode(i))


##########################################
def restore_model_with(path_to, name):
    path_to = path_to + '/' + name + '/'
    if not os.path.exists(path_to):
        raise FileNotFoundError

    objects = []
    for i in os.listdir(path_to):
        if i.__eq__(Settings.BUILD_MODEL):
            continue
        with open(path_to + i, "rb") as file:
            objects.append(jsonpickle.decode(
                file.read()))

    return load_model(path_to + Settings.BUILD_MODEL), \
           objects[0], objects[1], objects[2]


##########################################
# to_frames_video(app.Settings.TRAIN_VIDEO,
#                 app.Settings.TRAIN_FRAMES)
def to_frames_video(path_from, path_to, img_format='.jpg'):
    vid = cv2.VideoCapture('../' + path_from)

    path_to = '../' + path_to
    if not os.path.exists(path_to):
        os.makedirs(path_to)

    index = 0
    while True:
        # Extract images
        ret, frame = vid.read()
        if not ret:
            break
        # Saves images
        name = path_to + str(index) + img_format
        cv2.imwrite(name, frame)
        # next frame
        index += 1



