from itertools import product
import threading
import time

from sklearn.metrics import accuracy_score as accuracy
import numpy as np

from .file_utils import save_predictions, save_scores, save_times
from .progress_bar import ProgressBar


class GS:
    """ A Grid Search algorithm performed using Stratified KFolds

    Args:
        savedir (str): The absolute path to where results should be saved

    Attributes:
        cross_validator (sklearn.model_selection): The cross validation strategy
        savedir (str): The absolute path to where results should be saved
    """

    def __init__(self, savedir, cross_validator):
        self.selector = cross_validator
        self.savedir = savedir
        self._scores = []
        self._times = []
        self._pbar = None

    def tune(self, model, X, Y, **tuning_params):
        """ Fine tune on the model for a data with stratified K-fold

        Args:
            model (GrydModel): An object that implements GrydModel interface
            X (ndarray): The data points
            Y (ndarray): The expected classes
            **tuning_params (dict): The values for each parameter
        """
        self._pbar = ProgressBar(n_configs(tuning_params), 50, name='Tuning')
        self._configure_and_tune(model, X, Y, **tuning_params)
        save_scores(self.savedir, self._scores, tuning_params)
        save_times(self.savedir, self._times, tuning_params)

    def _configure_and_tune(self, model, X, Y, **tuning_params):
        for config in configurations(tuning_params):
            self._pbar.update()
            model.set_params(**config)
            scores = []
            times = []
            self._fit_and_test_timed(model, X, Y, config, scores, times)
            self._add_score(config, scores)
            self._add_time(config, times)

    def _fit_and_test(self, model, X, Y, config, scores, times):
        for train_index, test_index in self.selector.split(X, Y):
            Xtrain, Xtest = X[train_index], X[test_index]
            Ytrain, Ytest = Y[train_index], Y[test_index]

            model.fit(Xtrain, Ytrain)

            preds = model.predict(Xtest)
            scores.append(accuracy(preds, Ytest))
            save_predictions(self.savedir, config, preds, test_index, Ytest)

    def _fit_and_test_timed(self, model, X, Y, config, scores, times):
        for train_index, test_index in self.selector.split(X, Y):
            Xtrain, Xtest = X[train_index], X[test_index]
            Ytrain, Ytest = Y[train_index], Y[test_index]

            trntime, __ = _timeof(model.fit, Xtrain, Ytrain)
            tsttime, preds = _timeof(model.predict, Xtest)

            scores.append(accuracy(preds, Ytest))
            times.append(trntime)
            save_predictions(self.savedir, config, preds, test_index, Ytest)

    def _add_score(self, config, scores):
        line = [conf for conf in config.values()]
        mean = np.mean(scores) * 100
        std = np.std(scores) * 100
        line.extend([mean, std])
        self._scores.append(line)

    def _add_time(self, config, times):
        line = [conf for conf in config.values()]
        mean = np.mean(times) * 100
        std = np.std(times) * 100
        line.extend([mean, std])
        self._times.append(line)


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

