from sklearn import preprocessing


class PreprocessorParams:

    def __init__(
            self, backward=(0,), frame_y_trim=(230, 350),
            frame_x_trim=(140, 380), frame_scale=0.7):

        self._backward = backward
        self._frame_y_trim = frame_y_trim
        self._frame_x_trim = frame_x_trim
        self._frame_scale = frame_scale

    @property
    def backward(self):
        return self._backward

    @property
    def frame_y_trim(self):
        return self._frame_y_trim

    @property
    def frame_x_trim(self):
        return self._frame_x_trim

    @property
    def frame_scaler(self):
        return self._frame_scale


class ControllerParams:

    def __init__(self, baths=8, train_part=0.8,
                 samples=20400, step_vis=512):

        self._baths = baths
        self._train_part = train_part
        self._samples = samples
        self._step_vis = step_vis

    @property
    def baths(self):
        return self._baths

    @property
    def train_part(self):
        return self._train_part

    @property
    def samples(self):
        return self._samples

    @property
    def step_vis(self):
        return self._step_vis


class VisualHolder:
    def __init__(self):
        self._iters, self._costs = [], []
        self._evaluation = 1e+5

    def add(self, _iter, _cost):
        self._iters.append(_iter)
        self._costs.append(_cost)

    def set_evaluation(self, val):
        self._evaluation = val

    @property
    def evaluation(self):
        return self._evaluation

    @property
    def costs(self):
        return self._costs

    @property
    def iters(self):
        return self._iters
