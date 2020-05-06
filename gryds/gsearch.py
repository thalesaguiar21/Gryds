from itertools import product
import time

from sklearn.metrics import accuracy_score as accuracy
import numpy as np

from . import files
from .progress import ProgressBar
from . import base


def tune(model, mselector, X, Y, **tuning_params):
    """ Performs a grid search through the params using the model and a
    cross-validator

    Args:
        model: a mathmatical model that implements base.GrydsModel
        mselector: a cross validator from sklearn.model_selection
        X: the feature vectors
        Y: the vector labels
        **kwargs: the named parameters and its ranges

    Example:
        >>> from gryds import gsearch
        >>> model = sklearn.cluster.KMeans(n_cluster=2)
        >>> crossval = sklearn.model_selection.StratifiedKFold(3)
        >>> X, Y = sklearn.datasets.make_blobs()
        >>> gsearch.tune(model, crossval, X, Y, n_clusters=[2,4],
        >>>              max_iter=[100, 200])

        To change the directory

        >>> import gryds
        >>> gryds.confs.paths['save'] = 'path/to/dir/'
    """
    files.preconf(list(tuning_params))
    pbar = ProgressBar(n_configs(tuning_params), 50, name='Tuning')
    bestconf = base.ConfRes()
    for config in configurations(tuning_params):
        pbar.update()
        model.set_params(**config)
        results = _timed_fit_and_test(model, mselector, X, Y, config)
        currconf = base.ConfRes(results, config)
        bestconf = base.get_best(bestconf, currconf)
        files.save_results(results, config)
    print(bestconf)


def _timed_fit_and_test(model, mselector, X, Y, conf):
    result = base.Results()
    for trn_index, tst_index in mselector.split(X, Y):
        Xtrain, Xtest = X[trn_index], X[tst_index]
        Ytrain, Ytest = Y[trn_index], Y[tst_index]

        trntime, __ = _timeof(model.fit, Xtrain, Ytest)
        tsttime, preds = _timeof(model.predict, Xtest)

        score = accuracy(preds, Ytest)
        result.add(score, trntime, tsttime)

        files.save_predictions(conf, preds, tst_index, Ytest)
    return result


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
    """ Computes the number of combinations with the params """
    nconfs = 1
    for pvalues in tuning_params.values():
        nconfs *= len(pvalues)
    return nconfs

