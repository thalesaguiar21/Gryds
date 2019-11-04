from itertools import product

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score as accuracy
import numpy as np

from .file_utils import save_predictions


class GS:

    def __init__(self, nfolds, savedir):
        self.kfold = KFold(nfolds)
        self.savedir = savedir

    def tune(self, model, X, Y, **tunning_params):
        for config in configurations(tunning_params):
            model.set_params(**config)
            for train_index, test_index in self.kfold.split(X):
                Xtrain, Xtest = X[train_index], X[test_index]
                Ytrain, Ytest = Y[train_index], Y[test_index]

                model.fit(Xtrain)

                preds = model.predict(Xtest)
                mean_score = accuracy(preds, Ytest) / self.kfold.get_n_splits()
                save_predictions(self.savedir, config, preds, test_index, Ytest)


def configurations(tunning_parameters):
    pvalues = list(tunning_parameters.values())
    for param_set in product(*pvalues):
        yield dict(zip(tunning_parameters.keys(), param_set))

