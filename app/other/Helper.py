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
def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, Max y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.8", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.7, 0.7), **kw)


##########################################
def annot_min(x, y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    text= "x={:.3f}, Min y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.8", fc="w", ec="k", lw=0.4)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.4, 0.4), **kw)


##########################################
def annot_avr(y, ax=None):
    y_mean = np.mean(y)
    text= "Mean ={:.3f}".format(y_mean)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.8", fc="r", ec="k", lw=0.4)
    kw = dict(xycoords='data',textcoords="axes fraction",
              bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(0, 0), xytext=(0.1, 0.9), **kw)


##########################################
def clear_built_test(path_to, backward):
    file = path_to + '/' + Settings.BUILT_TEST

    if not os.path.exists(path_to):
        os.makedirs(path_to)

    if os.path.isfile(file):
        os.remove(file)
        with open(file, "wb") as file:
            np.savetxt(file, backward, delimiter="\n")


##########################################
def add_built_test(path_to, values):
    file = path_to + '/' + Settings.BUILT_TEST

    if not os.path.exists(path_to):
        os.makedirs(path_to)

    with open(file, "ab") as file:
        np.savetxt(file, np.round(
            values, 6), delimiter="\n")


##########################################
def save_plot(path_to, plot, prefix):
    if not os.path.exists(path_to):
        os.makedirs(path_to)
    plot.savefig(path_to + prefix + '.png')


##########################################
def backup_model_with(path_to, name, model, *args):
    path_to = path_to + '/' + Settings.MODELS\
              + '/' + name + '/'
    if not os.path.exists(path_to):
        os.makedirs(path_to)

    model.save(path_to + Settings.NAME_MODEL)
    for i in args:
        with open(path_to + type(i).__name__, "w+") as file:
            file.write(jsonpickle.encode(i))


##########################################
def restore_model_with(path_to, name):
    path_to = path_to + '/' + Settings.MODELS \
              + '/' + name + '/'
    if not os.path.exists(path_to):
        raise FileNotFoundError

    p_params, c_params, visual = \
        None, None, None

    for i in os.listdir(path_to):
        if i.__eq__(Settings.NAME_MODEL):
            continue

        with open(path_to + i, "rb") as file:
            try:
                decoded = jsonpickle.decode(
                    file.read())
                if type(decoded).__name__.__eq__(
                        'PreprocessorParams'):
                    p_params = decoded
                elif type(decoded).__name__.__eq__(
                        'ControllerParams'):
                    c_params = decoded
                else:
                    visual = decoded
            except UnicodeDecodeError:
                pass

    return load_model(path_to + Settings.NAME_MODEL),\
           p_params, c_params, visual


##########################################
# to_frames_video('../../' + Settings.TEST_VIDEO,
#                 '../../' + Settings.TEST_FRAMES)
#
# ffmpeg -i source/test.mp4 -y -an -f image2 /
#      -r 20 frames-t/%01d.jpg
def to_frames_video(path_from, path_to, img_format='.jpg'):
    vid = cv2.VideoCapture(path_from)

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


