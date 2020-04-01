import numpy as np

from . import confs


class GrydModel:
    """Base interface used by the gsearch module"""

    def fit(self, X, Y):
        """Train the model with the given data"""
        pass

    def predict(self, data):
        """Predict the classes of the given samples"""
        pass

    def set_params(self, **kwargs):
        """Set tunning parameters for this model"""
        pass


class Results:
    """A simple class to hold several results of each type"""

    def __init__(self):
        self.scores = []
        self.traintimes = []
        self.testtimes = []

    def add(self, acc, train, test):
        self.scores.append(acc)
        self.traintimes.append(train)
        self.testtimes.append(test)


def to_saveformat(result):
    """Converts each result attribute to hold [mean, stdev] and applies any
    scaling factor defined in 'confs' module

    """
    result.scores = _make_mean_std(result.scores, 100)
    timeunit = confs.get_timeunit()
    result.traintimes = _make_mean_std(result.traintimes, timeunit)
    result.testtimes = _make_mean_std(result.testtimes, timeunit)


def _make_mean_std(field, scale):
    mean = np.mean(field) * scale
    std = np.std(field) * scale
    return mean, std

