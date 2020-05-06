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


def build_zero_res():
    zero = Results()
    zero.scores = [0, 0]
    zero.traintimes = [0, 0]
    zero.testtimes = [0, 0]
    return zero


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


class ConfRes:

    def __init__(self, result=None, conf=None):
        result = build_zero_res() if result is None else result
        self.conf = conf

        tmunt = confs.get_timeunit()
        accavg, accstd = _make_mean_std(result.scores, 100)
        trntimeavg, trntimestd = _make_mean_std(result.traintimes, tmunt)
        tsttimeavg, tsttimestd = _make_mean_std(result.testtimes, tmunt)

        self.accavg = accavg
        self.accstd = accstd
        self.trntimeavg = trntimeavg
        self.trntimestd = trntimestd
        self.tsttimeavg = tsttimeavg
        self.tsttimestd = tsttimestd

    def __str__(self):
        accview = f"Accuracy:\t{self.accavg:3.2f}% +- {self.accstd:3.2f}%"
        trainview = f"Training time:\t{self.trntimeavg:3.2f}s +- {self.trntimestd:3.2f}s"
        testview = f"Test time:\t{self.tsttimeavg:3.2f}s +- {self.tsttimestd:3.2f}s"

        return f"{self.conf}\n{accview}\n{trainview}\n{testview}"

    def __repr__(self):
        return self.__str__()


def get_best(conf1, conf2):
    best = conf1
    if conf1.accavg == conf2.accavg:
        if conf1.accstd > conf2.accstd:
            best = conf2
    elif conf2.accavg > conf1.accavg:
        best = conf2
    return best

