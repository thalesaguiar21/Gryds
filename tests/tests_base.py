import os
import unittest

import numpy as np

from .context import gryds
from gryds import base


class TestsBase(unittest.TestCase):

    def tearDown(self):
        gryds.confs.metrics['timeunit'] = 'sec'

    def _build_result(self):
        result = base.Results()
        result.scores = [i/10 for i in range(10)]
        result.traintimes = [i/10 for i in range(10)]
        result.testtimes = [i/10 for i in range(10)]
        return result

    def test_saveformat_acc(self):
        result = self._build_result()
        base.as_saveformat(result)
        for accuracy in result.scores:
            self.assertLessEqual(accuracy, 100)
            self.assertGreaterEqual(accuracy, 0)

    def test_saveformat_trntime_sec(self):
        gryds.confs.metrics['timeunit'] = 'sec'
        self._test_saveformat(1)

    def test_saveformat_trntime_mili(self):
        gryds.confs.metrics['timeunit'] = 'mili'
        self._test_saveformat(1e-3)

    def test_saveformat_trntime_nano(self):
        gryds.confs.metrics['timeunit'] = 'nano'
        self._test_saveformat(1e-9)

    def _test_saveformat(self, scale):
        result = self._build_result()
        meantime = np.mean(result.traintimes) * scale
        stdtime = np.std(result.traintimes) * scale
        sres = base.as_saveformat(result)
        self.assertAlmostEqual(meantime, sres.traintimes[0])
        self.assertAlmostEqual(stdtime, sres.traintimes[1])


class TestsResults(unittest.TestCase):

    def test_add(self):
        result = base.Results()
        result.add(3, 4, 5)
        
        self.assertEqual(result.scores, [3])
        self.assertEqual(result.traintimes, [4])
        self.assertEqual(result.testtimes, [5])

        result.add(None, 0.1, 'test')

        self.assertEqual(result.scores, [3, None])
        self.assertEqual(result.traintimes, [4, 0.1])
        self.assertEqual(result.testtimes, [5, 'test'])
