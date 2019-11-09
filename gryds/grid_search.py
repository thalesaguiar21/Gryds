from itertools import product

from sklearn.metrics import accuracy_score as accuracy
import numpy as np

from .file_utils import save_predictions, save_scores


class GS:
    """ A Grid Search algorithm performed using Stratified KFolds

    Args:
        savedir (str): The absolute path to where results should be saved

    Attributes:
        cross_validator (sklearn.model_selection): The cross validation strategy
        savedir (str): The absolute path to where results should be saved
    """

    def __init__(self, savedir, cross_validator):
        self.cross_validator = cross_validator
        self.savedir = savedir

    def tune(self, model, X, Y, **tuning_params):
        """ Fine tune on the model for a data with stratified K-fold

        Args:
            model (GrydModel): An object that implements GrydModel interface
            X (ndarray): The data points
            Y (ndarray): The expected classes
            **tuning_params (dict): The values for each parameter
        """
        self._configure_and_tune(model, X, Y, **tuning_params)

    def _configure_and_tune(self, model, X, Y, **tuning_params):
        for config in configurations(tuning_params):
            model.set_params(**config)
            scores = []
            self._fit_and_test(model, X, Y, config, scores)
            save_scores(self.savedir, config, scores)

    def _fit_and_test(self, model, X, Y, config, scores):
        for train_index, test_index in self.cross_validator.split(X, Y):
            Xtrain, Xtest = X[train_index], X[test_index]
            Ytrain, Ytest = Y[train_index], Y[test_index]

            model.fit(Xtrain, Ytrain)

            preds = model.predict(Xtest)
            scores.append(accuracy(preds, Ytest))
            save_predictions(self.savedir, config, preds, test_index, Ytest)


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

