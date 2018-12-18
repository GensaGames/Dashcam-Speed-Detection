from collections import deque


class PreprocessorParams:

    def __init__(
            self, backward=(0,), frame_y_trim=(230, 350),
            frame_x_trim=(140, 380), area_float=10,
            frame_scale=0.7):

        self._backward = backward
        self._frame_y_trim = frame_y_trim
        self._frame_x_trim = frame_x_trim
        self._frame_scale = frame_scale
        self._area_float = area_float

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

    @property
    def area_float(self):
        return self._area_float


class ControllerParams:

    def __init__(self, name, baths=8, train_part=0.8, epochs=10,
                 samples=20400, step_vis=512):

        self._name = name
        self._baths = baths
        self._train_part = train_part
        self._epochs = epochs
        self._samples = samples
        self._step_vis = step_vis

    @property
    def name(self):
        return self._name

    @property
    def baths(self):
        return self._baths

    @property
    def epochs(self):
        return self._epochs

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
        self._iters, self._costs = \
            deque(maxlen=1000), deque(maxlen=1000)
        self._evaluations = deque(maxlen=1000)

    def add_iter(self, _iter, _cost):
        self._iters.append(_iter)
        self._costs.append(_cost)

    def add_evaluation(self, val):
        self._evaluations.append(val)

    @property
    def evaluations(self):
        return self._evaluations

    @property
    def costs(self):
        return self._costs

    @property
    def iters(self):
        return self._iters
