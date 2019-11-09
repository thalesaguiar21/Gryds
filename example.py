import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

from gryds.grid_search import GS

path = os.path.abspath('tests/blobs.txt')
data = np.loadtxt(path)
X, Y = data[:, :-1], data[:, -1]
cross_validator = StratifiedKFold(3)

gs = GS(os.path.abspath('tests/'), cross_validator)
gs.tune(KMeans(n_clusters=2), X, Y, n_clusters=[i*2 for i in range(1, 301)],
        max_iter=[i*100 for i in range(1, 11)])

