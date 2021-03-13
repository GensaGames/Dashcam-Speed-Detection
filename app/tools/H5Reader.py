import cv2
import h5py

from app import Settings


class H5Reader:

    def __init__(self, params):
        self.params = params

    def showImages(self):
        with h5py.File(self.params.video_p, "r") as f:

            for i in range(2000, int(1e+4)):
                frame = f['X'][i]
                
                cv2.imshow('Dataset Image', frame[-1])
                cv2.waitKey(0)


class H5ReaderParams:
    def __init__(self, video, log):
        self.video_p = video
        self.log_p = log


#####################################
if __name__ == "__main__":
    H5Reader(H5ReaderParams(
        Settings.EXTRA_DATA_V1,
        Settings.EXTRA_DATA_L1,
    )).showImages()
