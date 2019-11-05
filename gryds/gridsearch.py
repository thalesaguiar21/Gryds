from itertools import product

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score as accuracy
import numpy as np

from .file_utils import save_predictions, save_scores


class GS:
    """ A Grid Search algorithm performed using Stratified KFolds

    Args:
        nfolds (int): The number of splits for the data
        savedir (str): The absolute path to where results should be saved

    Attributes:
        kfold (sklearn.model_selection.KFold): The stratified KFold object
        savedir (str): The absolute path to where results should be saved
    """

    def __init__(self, nfolds, savedir):
        self.kfold = KFold(nfolds)
        self.savedir = savedir

    def tune(self, model, X, Y, **tunning_params):
        """ Fine tune on the model for a data with stratified K-fold

        Args:
            model (GrydModel): An object that implements GrydModel interface
            X (ndarray): The data points
            Y (ndarray): The expected classes
            **tunning_parameters (dict): The values for each parameter
        """
        for config in configurations(tunning_params):
            model.set_params(**config)
            mean_score = 0
            for train_index, test_index in self.kfold.split(X):
                Xtrain, Xtest = X[train_index], X[test_index]
                Ytrain, Ytest = Y[train_index], Y[test_index]

                model.fit(Xtrain)

                preds = model.predict(Xtest)
                mean_score += accuracy(preds, Ytest) / self.kfold.get_n_splits()
                save_predictions(self.savedir, config, preds, test_index, Ytest)
            save_scores(self.savedir, config, mean_score)


def configurations(tunning_parameters):
    """ Create every combination for the given dictionary values

    Args:
        tunning_parameters: dictionary

    Yields:
        config: The  configuration mapping parameter name to value

    Examples:
        >>> print(list(configurations({'a':[2, 3], 'b':[4, 5]})))
        [{'a':2, 'b':4}, {'a':2, 'b':5}, {'a':3, 'b':4}, {'a':3, 'b':5}]
    """
    pvalues = list(tunning_parameters.values())
    for param_set in product(*pvalues):
        yield dict(zip(tunning_parameters.keys(), param_set))

