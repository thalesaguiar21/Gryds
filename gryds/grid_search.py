from itertools import product
import time

from sklearn.metrics import accuracy_score as accuracy
import numpy as np

from .file_utils import save_predictions, save_scores, save_times
from .progress_bar import ProgressBar


class _Results:

    def __init__(self):
        self.scores = []
        self.traintimes = []
        self.testtimes = []

    def add(scr, train, test):
        self.scores += scr
        self.traintimes += train
        self.testtimes += test


def tune(model, X, Y, mselector, **tuning_params):
    _pbar = ProgressBar(n_configs(tuning_params), 50, name='Tuning')
    for config in configurations(tuning_params):
        _pbar.update()
        model.set_params(**config)
        results = timed_fit_and_test(model, X, Y)


def _timed_fit_and_test(model, mselector, X, Y):
    result = _Results()
    for trn_index, tst_index in mselector.split(X, Y):
        Xtrain, Xtest = X[trn_index], X[tst_index]
        Ytrain, Ytest = Y[trn_index], Y[tst_index]

        trntime, __ = timeof(model.fit, Xtrain)
        tsttime, preds = timeof(model.predict, Xtest)

        score = accuracy(preds, Ytest)
        results.add(score, trntime, tsttime)

        save_predictions(config, preds, test_index, Ytest)


def _make_mean_std(results):
    for field in dataclass.as_tuple():
        smean = np.mean(field)
        sstd = np.std(field)
        yield smean, sstd


def _timeof(func, *args):
    start = time.perf_counter()
    fout = func(*args)
    end = time.perf_counter()
    elapsed = end - start
    return elapsed, fout


def configurations(tuning_params):
    """ Create every combination for the given dictionary values

    Args:
        tuning_params: dictionary

    Yields:
        config: The  configuration mapping parameter name to value

    Examples:
        >>> print(list(configurations({'a':[2, 3], 'b':[4, 5]})))
        [{'a':2, 'b':4}, {'a':2, 'b':5}, {'a':3, 'b':4}, {'a':3, 'b':5}]
    """
    pvalues = list(tuning_params.values())
    for param_set in product(*pvalues):
        yield dict(zip(tuning_params.keys(), param_set))


def n_configs(tuning_params):
    nconfs = 1
    for pvalues in tuning_params.values():
        nconfs *= len(pvalues)
    return nconfs

