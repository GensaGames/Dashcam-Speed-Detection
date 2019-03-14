import os

PROJECT = os.path.dirname(
    os.path.realpath(__file__)) + '/../'
RESOURCE = PROJECT + 'resource/'

TRAIN_VIDEO = RESOURCE + 'source/train.mp4'
TRAIN_FRAMES = RESOURCE + 'frames/'
TRAIN_Y = RESOURCE + 'source/train.txt'
TRAIN_FRAMES_STOP = RESOURCE + 'frames-stop/'

TEST_VIDEO = RESOURCE + 'source/test.mp4'
TEST_FRAMES = RESOURCE + 'frames-t/'
NAME_LOGS = 'logs.txt'

BUILD = PROJECT + 'build/'
BUILT_TEST = 'optical-3d-cnn-v8-source.txt'
BUILT_TEST_PR1 = 'test-pp1.txt'

MODELS = 'models'
NAME_MODEL = 'model-view.h5'
NAME_MODEL_PLOT = 'epochs-plot.png'


