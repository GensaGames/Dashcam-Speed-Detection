import os

import cv2
import jsonpickle
import numpy as np
import jsonpickle.ext.numpy as jsonpickle_numpy


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
def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x ** 2


##########################################
def to_timeline(value):
    sqrt = int(np.sqrt(value.shape[0]))
    return value.reshape(sqrt, sqrt,
                         value.shape[1])


##########################################
def save_state_with(path_to, plot, prefix, learner, p_params, visual):
    if not os.path.exists(path_to):
        os.makedirs(path_to)

    name = path_to + prefix + '-' + str(type(learner).__name__) + '-LookBack-' \
           + str(p_params.backward) + '-Layers-' + str('~') \
           + '-LR-' + str('~') + '-Units-' + str('~')

    # Remove UnSerializable field from Preprocessor
    p_params_dict = p_params.__dict__
    del p_params_dict['_feature_scaler']

    with open(name + '.txt', "w") as file:
        file.write("{}\n\n{}".format(to_json_state(
            p_params_dict), to_json_state(visual)))


##########################################
def to_json_state(state):
    jsonpickle_numpy.register_handlers()
    json = jsonpickle.encode(state)
    return json


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



