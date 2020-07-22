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
        image2=Sequential([
            Sometimes(
                0.8, CoarseDropout(
                    p=(0.05, 0.3), size_percent=(0.1, 0.6))
            ),
        ]),
        area_float=6,
        features_dropout=0.1
    )


def get_new_validation():
    return AugmenterSettings(
        image=Sequential([
            AllChannelsCLAHE(),
        ]),
        image2=Sequential([
        ]),
        area_float=0,
        features_dropout=0.0
    )


class AugmenterSettings:

    # noinspection PyDefaultArgument
    def __init__(
            self, image, image2, area_float=0,
            features_dropout=0.0
    ):
        self.image = image
        self.image2 = image2
        self.area_float = area_float
        self.features_dropout = features_dropout


#####################################
if __name__ == "__main__":
    assert get_new_training() is not None
