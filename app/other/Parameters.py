from collections import deque


class ControllerParams:

    def __init__(
            self, name, baths=20, train_part=0.7,
            epochs=1, step_vis=10
    ):

        self._name = name
        self._baths = baths
        self._train_part = train_part
        self._epochs = epochs
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
    def step_vis(self):
        return self._step_vis


class VisualHolder:
    def __init__(self):
        self._trainings = deque()
        self._validation = deque()

    def add_training_point(self, _training):
        self._trainings.append(_training)

    def add_validation_point(self, val):
        self._validation.append(val)

    @property
    def points_validation(self):
        return self._validation

    @property
    def points_training(self):
        return self._trainings
