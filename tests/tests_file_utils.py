import os
import unittest

from .context import gryds
from gryds import file_utils as futils


PATH = os.path.abspath('tests/test_dir/')


class TestsFindFiles(unittest.TestCase):

    def test_nfiles_txt(self):
        files = futils.find_files(PATH, 'txt')
        self.assertEqual(len(files), 4)

    def test_nfiles_wav(self):
        files = futils.find_files(PATH, 'wav')
        self.assertEqual(len(files), 1)

    def test_nfiles_lab(self):
        files = futils.find_files(PATH, 'lab')
        self.assertEqual(len(files), 0)

    def test_unknown_path(self):
        path = ''
        files = futils.find_files(path, 'txt')
        self.assertEqual(len(files), 0)

    def test_upper_extension(self):
        files = futils.find_files(PATH, 'TXT')
        self.assertEqual(len(files), 4)

    def test_mixed_case_extension(self):
        files = futils.find_files(PATH, 'TxT')
        self.assertEqual(len(files), 4)


class TestsFindTxt(unittest.TestCase):

    def test_nfiles(self):
        files = futils.find_files(PATH, 'txt')
        txt_files = futils.find_txt_files(PATH)
        self.assertEqual(len(files), len(txt_files))


class TestsFindWav(unittest.TestCase):

    def test_nfiles(self):
        files = futils.find_files(PATH, 'wav')
        wav_files = futils.find_wav_files(PATH)
        self.assertEqual(len(files), len(wav_files))


class TestsPredName(unittest.TestCase):

    def test_pred_name(self):
        config = {'a':'2', 'b':'3'}
        path = os.path.abspath('tests/')
        expectedname = f"{path}/a_2_b_3.preds"
        fname = futils.make_pred_name(path, config)
        self.assertEqual(expectedname, fname)

