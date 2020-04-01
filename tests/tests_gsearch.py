import os
import unittest

from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
import numpy as np

from .context import gryds
from gryds import gsearch as gs


PATH = os.path.abspath('tests/blobs.txt')
EXTENSION = gryds.confs.get_extension()


class TestsGS(unittest.TestCase):

    def tearDown(self):
        path = os.path.abspath('tests/')
        dircontent = os.listdir(path)
        files = [fname for fname in dircontent if os.path.isfile(path + fname)]
        for fname in files:
            if fname.endswith(EXTENSION):
                os.remove(path + fname)

    def test_run(self):
        cross_validator = StratifiedKFold(3)
        data = np.loadtxt(PATH)
        X, Y = data[:, :-1], data[:, -1]
        gs.tune(KMeans(n_clusters=2), cross_validator, X, Y, n_clusters=[2, 4],
                max_iter=[100, 200])


class TestsConfs(unittest.TestCase):

    def test_timeunit_default(self):
        gryds.confs.metrics['timeunit'] = 'aehoo'
        default = gryds.confs.get_timeunit()
        self.assertEqual(1, default)

    def test_timeunit_mili(self):
        gryds.confs.metrics['timeunit'] = 'mili'
        milisec = gryds.confs.get_timeunit()
        self.assertEqual(1e-3, milisec)

    def test_timeunit_nano(self):
        gryds.confs.metrics['timeunit'] = 'nano'
        nanosec = gryds.confs.get_timeunit()
        self.assertEqual(1e-9, nanosec)



class TestsConfiguration(unittest.TestCase):

    def test_n_configs(self):
        params = {'a':[1, 2, 3], 'b':['a', 'c'], 'd':[33, 12]}
        configs = list(gs.configurations(params))
        self.assertEqual(12, len(configs))

    def test_configs_function(self):
        params = {'a':[1, 2, 3], 'b':['a', 'c'], 'd':[33, 12]}
        nconfigs = gs.n_configs(params)
        self.assertEqual(12, nconfigs)

