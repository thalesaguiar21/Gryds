import os
import unittest

from sklearn.cluster import KMeans
import numpy as np

from .context import gryds
from gryds.gridsearch import GS, configurations


PATH = os.path.abspath('tests/blobs.txt')


class TestsGS(unittest.TestCase):

    def test_run(self):
        gs = GS(3)
        data = np.loadtxt(PATH)
        X, Y = data[:, :-1], data[:, -1]
        gs.tune(KMeans(n_clusters=2), X, Y, n_clusters=[2, 3, 4],
                max_iter=[100, 200, 300], algorithm=['auto', 'full', 'elkan'])


class TestsConfiguration(unittest.TestCase):

    def test_n_configs(self):
        params = {'a':[1, 2, 3], 'b':['a', 'c'], 'd':[33, 12]}
        configs = list(configurations(params))
        self.assertEqual(12, len(configs))

