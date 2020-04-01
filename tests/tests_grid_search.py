import os
import unittest

from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
import numpy as np

from .context import gryds
from gryds import grid_search as gs


PATH = os.path.abspath('tests/blobs.txt')


class TestsGS(unittest.TestCase):

    def tearDown(self):
        path = os.path.abspath('tests/')
        dircontent = os.listdir(path)
        files = [fname for fname in dircontent if os.path.isfile(path + fname)]
        for fname in files:
            if fname.endswith('.pred'):
                os.remove(path + pred_file)
        os.remove(f"{path}/scores.txt")
        os.remove(f"{path}/times.txt")

    def test_run(self):
        cross_validator = StratifiedKFold(3)
        data = np.loadtxt(PATH)
        X, Y = data[:, :-1], data[:, -1]
        gs.tune(KMeans(n_clusters=2), cross_validator, X, Y, n_clusters=[2, 4],
                max_iter=[100, 200])


class TestsConfiguration(unittest.TestCase):

    def test_n_configs(self):
        params = {'a':[1, 2, 3], 'b':['a', 'c'], 'd':[33, 12]}
        configs = list(gs.configurations(params))
        self.assertEqual(12, len(configs))

    def test_configs_function(self):
        params = {'a':[1, 2, 3], 'b':['a', 'c'], 'd':[33, 12]}
        nconfigs = gs.n_configs(params)
        self.assertEqual(12, nconfigs)

