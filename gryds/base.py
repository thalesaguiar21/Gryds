import numpy as np

from . import confs


class GrydModel:

    def fit(self, X, Y):
        ''' Train the model with the given data '''
        pass

    def predict(self, data):
        ''' Predict the classes of the given samples '''
        pass

    def set_params(self, **kwargs):
        ''' Set tunning parameters for this model '''
        pass


class Results:

    def __init__(self):
        self.scores = []
        self.traintimes = []
        self.testtimes = []

    def add(self, scr, train, test):
        self.scores.append(scr)
        self.traintimes.append(train)
        self.testtimes.append(test)


def to_saveformat(result):
    result.scores = _make_mean_std(result.scores, 100)
    timeunit = confs.get_timeunit()
    result.traintimes = _make_mean_std(result.traintimes, timeunit)
    result.testtimes = _make_mean_std(result.testtimes, timeunit)


def _make_mean_std(field, scale):
    mean = np.mean(field) * scale
    std = np.std(field) * scale
    return mean, std

