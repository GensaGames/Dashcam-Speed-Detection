from imgaug.augmenters import CoarseDropout, AllChannelsCLAHE, Fliplr, MultiplyAndAddToBrightness, Multiply, \
    WithBrightnessChannels, CSPACE_HSV, CSPACE_Lab, Add
from imgaug.augmenters import Sequential
from imgaug.augmenters import Sometimes


# Also check with:
# AllChannelsHistogramEqualization()
def get_new_training():
    return Sequential([
        Fliplr(0.5),
        WithBrightnessChannels(
            Add((-50, 50)), to_colorspace=[CSPACE_Lab, CSPACE_HSV])
    ])


def get_new_validation():
    return Sequential([
        WithBrightnessChannels(
            Add((10, 30)), to_colorspace=[CSPACE_Lab, CSPACE_HSV])
    ])


def get_new_empty():
    return Sequential([
    ])
