from itertools import product

from sklearn.model_selection import KFold
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score as accuracy
import numpy as np


class GS:

    def __init__(self, nfolds):
        self.kfold = KFold(nfolds)

    def tune(self, model, X, Y, **tunning_params):
        for config in configurations(tunning_params):
            model.set_params(**config)
            for train_index, test_index in self.kfold.split(X):
                Xtrain, Xtest = X[train_index], X[test_index]
                Ytrain, Ytest = Y[train_index], Y[test_index]

                model.fit(Xtrain)

                predictions = model.predict(Xtest)
                score = accuracy(predictions, Ytest) / self.kfold.get_n_splits()



def configurations(tunning_parameters):
    pvalues = list(tunning_parameters.values())
    for param_set in product(*pvalues):
        yield dict(zip(tunning_parameters.keys(), param_set))


if __name__ == '__main__':
    gs = GS(3)
    data = np.loadtxt('tests/blobs.txt')
    X, Y = data[:, :-1], data[:, -1]
    gs.tune(KMeans(n_clusters=2), X, Y, n_clusters=[2, 3, 4],
            max_iter=[100, 200, 300], algorithm=['auto', 'full', 'elkan'])
