import os
import unittest

from .context import gryds
from gryds import file_utils as futils


class TestsPredName(unittest.TestCase):

    def test_pred_name(self):
        config = {'a':'2', 'b':'3'}
        path = os.path.abspath('tests/')
        extension = gryds.confs.get_extension()
        expectedname = f"{path}/a_2_b_3{extension}"
        fname = futils.make_pred_name(path, config)
        self.assertEqual(expectedname, fname)

