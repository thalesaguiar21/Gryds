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


def as_saveformat(result):
    """Converts each result attribute to hold [mean, stdev] and applies any
    scaling factor defined in 'confs' module

    """
    sres = Results()
    sres.scores =  _make_mean_std(result.scores, 100)
    timeunit = confs.get_timeunit()
    sres.traintimes = _make_mean_std(result.traintimes, timeunit)
    sres.testtimes = _make_mean_std(result.testtimes, timeunit)
    return sres


def _make_mean_std(field, scale):
    mean = np.mean(field) * scale
    std = np.std(field) * scale
    return mean, std


class ConfRes:

    def __init__(self, result=None, conf=None):
        result = build_zero_res() if result is None else result
        self.conf = conf
        self.result = as_saveformat(result)

    def accs(self):
        return self.result.scores

    def traintimes(self):
        return self.result.traintimes

    def testtimes(self):
        return self.result.testtimes

    def __str__(self):
        accavg, accstd = self.accs()
        accview = f"Accuracy:\t{accavg:3.2f}% +- {accstd:3.2f}%"
        trntimeavg, trntimestd = self.traintimes()
        trainview = f"Training time:\t{trntimeavg:3.2e}s +- {trntimestd:3.2e}s"
        tsttimeavg, tsttimestd = self.testtimes()
        testview = f"Test time:\t{tsttimeavg:3.2e}s +- {tsttimestd:3.2e}s"

        return f"{self.conf}\n{accview}\n{trainview}\n{testview}"

    def __repr__(self):
        return self.__str__()


def get_best(conf1, conf2):
    best = conf1
    m1, s1 = conf1.accs()
    m2, s2 = conf2.accs()

    if m1 == m2:
        if s1 > s2:
            best = conf2
    elif m2 > m1:
        best = conf2
    return best

