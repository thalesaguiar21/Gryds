import os
import unittest

from sklearn.cluster import KMeans
import numpy as np

from .context import gryds
from gryds.gridsearch import GS, configurations, make_pred_name


PATH = os.path.abspath('tests/blobs.txt')


class TestsGS(unittest.TestCase):

    def test_run(self):
        gs = GS(3, os.path.abspath('tests/'))
        data = np.loadtxt(PATH)
        X, Y = data[:, :-1], data[:, -1]
        gs.tune(KMeans(n_clusters=2), X, Y, n_clusters=[2, 4],
                max_iter=[100, 200])


class TestsConfiguration(unittest.TestCase):

    def test_n_configs(self):
        params = {'a':[1, 2, 3], 'b':['a', 'c'], 'd':[33, 12]}
        configs = list(configurations(params))
        self.assertEqual(12, len(configs))


class TestsPredName(unittest.TestCase):

    def setUp(self):
        self.gs = GS(3, os.path.abspath('tests/'))

    def test_name(self):
        conf = {'a':'2', 'b':'x'}
        fname = make_pred_name(self.gs.savedir, conf)
        realname = os.path.abspath('tests/') + '/a_2_b_x.preds'
        self.assertEqual(realname, fname)
