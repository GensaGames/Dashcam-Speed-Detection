from imgaug.augmenters import CoarseDropout, AllChannelsCLAHE, Fliplr
from imgaug.augmenters import Sequential
from imgaug.augmenters import Sometimes


# Also check with:
# AllChannelsHistogramEqualization()
def get_new_training():
    return AugmenterSettings(
        image=Sequential([
            AllChannelsCLAHE(),
            Fliplr(0.5),
        ]),
        area_float=6,
        features_dropout=0.1
    )


def get_new_validation():
    return AugmenterSettings(
        image=Sequential([
            AllChannelsCLAHE(),
        ]),
        area_float=0,
        features_dropout=0.0
    )


class AugmenterSettings:

    # noinspection PyDefaultArgument
    def __init__(
            self, image, area_float=0,
            features_dropout=0.0
    ):
        self.image = image
        self.area_float = area_float
        self.features_dropout = features_dropout


#####################################
if __name__ == "__main__":
    assert get_new_training() is not None
