import os

PROJECT = os.path.dirname(
    os.path.realpath(__file__)) + '/../'
RESOURCE = PROJECT + 'res/'

TRAIN_VIDEO = RESOURCE + 'source/train.mp4'
TRAIN_FRAMES = RESOURCE + 'frames/'
TRAIN_Y = RESOURCE + 'source/train.txt'
TRAIN_FRAMES_STOP = RESOURCE + 'frames-stop/'

TEST_VIDEO = RESOURCE + 'source/test.mp4'
TEST_FRAMES = RESOURCE + 'frames-t/'
NAME_LOGS = 'logs.txt'

BUILD = PROJECT + 'build/'
BUILT_TEST_PR1 = 'test-pp1.txt'
BUILT_TEST_FORMAT = '.txt'

MODELS = 'models'
NAME_MODEL = 'model-view.h5'
NAME_MODEL_PLOT = 'epochs-plot.png'
NAME_STRUCTURE = 'structure.png'

PREFIX_STOP_SIZE = int(10.e+4)


WRITE_VIDEO_PATH = RESOURCE + 'videos-o/'
WRITE_VIDEO_1 = WRITE_VIDEO_PATH + 'Bilkamera-F.mp4'
WRITE_VIDEO_2 = WRITE_VIDEO_PATH + 'y2mate.com-F.mp4'

EXTRA_DATA = PROJECT + '../../Data/CommaAi/'
EXTRA_DATA_V1 = EXTRA_DATA + 'camera/2016-01-30--11-24-51.h5'
EXTRA_DATA_L1 = EXTRA_DATA + 'log/2016-01-30--11-24-51.h5'


