import numpy as np


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


def convert_to_mean_std(result):
    result.scores = _make_mean_std(result.scores)
    result.traintimes = _make_mean_std(result.traintimes)
    result.testtimes = _make_mean_std(result.testtimes)


def _make_mean_std(field):
    mean = np.mean(field)
    std = np.std(field)
    return mean, std

