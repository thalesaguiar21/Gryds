import os
import sys

import numpy as np


PRED_EXTS = '.preds'
SCORE_EXTS = '.scores'


def find_files(path, extension):
    ''' Given an absolute path, return the absolute path and name of every
    file in the given path and its subdirectories.
    '''
    files = []
    extension = '.' + extension.lower()
    for base, directories, fnames in os.walk(path):
        for fname in fnames:
            if fname.lower().endswith(extension):
                files.append(os.path.join(base, fname))
    return files


def find_txt_files(path):
    return find_files(path, 'txt')


def find_wav_files(path):
    return find_files(path, 'wav')


def save_predictions(path, config, predictions, sample_indexes, yreal):
    fname = make_pred_name(path, config)
    results = np.vstack((predictions, sample_indexes, yreal)).T
    np.savetxt(fname, results, '%3.7f\t%4i\t%3.7f')


def _make_conf_path(dir_, config):
    conf_name = []
    for key in config.keys():
        conf_name.append(key)
        conf_name.append(str(config[key]))
    fname = '_'.join(conf_name)
    return f"{dir_}/{fname}"


def make_pred_name(dir_, config):
    path = _make_conf_path(dir_, config)
    return f"{path}{PRED_EXTS}"


def make_score_name(dir_, config):
    path = _make_conf_path(dir_, config)
    return f"{path}{SCORE_EXTS}"

