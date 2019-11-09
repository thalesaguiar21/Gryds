import os
import unittest

from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
import numpy as np

from .context import gryds
from gryds.grid_search import GS, configurations, n_configs
from gryds.file_utils import find_files, PRED_EXTS


PATH = os.path.abspath('tests/blobs.txt')


class TestsGS(unittest.TestCase):

    def tearDown(self):
        path = os.path.abspath('tests/')
        for pred_file in find_files(path, 'preds'):
            os.remove(pred_file)
        os.remove(f"{path}/scores.txt")

    def test_run(self):
        cross_validator = StratifiedKFold(3)
        gs = GS(os.path.abspath('tests/'), cross_validator)
        data = np.loadtxt(PATH)
        X, Y = data[:, :-1], data[:, -1]
        gs.tune(KMeans(n_clusters=2), X, Y, n_clusters=[2, 4],
                max_iter=[100, 200])


class TestsConfiguration(unittest.TestCase):

    def test_n_configs(self):
        params = {'a':[1, 2, 3], 'b':['a', 'c'], 'd':[33, 12]}
        configs = list(configurations(params))
        self.assertEqual(12, len(configs))

    def test_configs_function(self):
        params = {'a':[1, 2, 3], 'b':['a', 'c'], 'd':[33, 12]}
        nconfigs = n_configs(params)
        self.assertEqual(12, nconfigs)

