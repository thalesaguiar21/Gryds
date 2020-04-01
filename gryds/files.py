import os
import sys

import numpy as np

from . import confs
from . import base


SAVEDIR = confs.get_savedir()
EXTENSION = confs.get_extension()
_FNAMES = ['trntimes', 'tsttimes', 'accuracies']
_TRAINTIMES = 0
_TESTTIMES = 1
_ACCURACIES = 2


def preconf_files(config):
    for scorefile in _FNAMES:
        path = SAVEDIR + scorefile + EXTENSION
        cols = config[:] + ['mean', 'std']
        with open(path, 'w') as file:
            header = ','.join(cols) + '\n'
            file.write(header)


def save_predictions(config, predictions, sample_indexes, yreal):
    """ Create a file mapping prediction, sample index, and expected output

    Args:
        path (str): The path to which the file must be stored
        config (dict): A configuration given in a dict

    Exapmle:
        >>> path = /path/to/sotre/
        >>> config = {'a':2, 'b':4}
        >>> predictions = [1, 2]
        >>> sample_indexes = [35, 32]
        >>> yreal = [1, 3]
        >>> save_predictions(path, config, predictions, sample_indexes, yreal)
        >>> print(open(path + 'a_2_b_4.txt').read())
        35  1   1
        32  2   3
    """
    fname = make_pred_name(SAVEDIR, config)
    results = np.vstack((sample_indexes, predictions, yreal)).T
    with open(fname, 'w+') as pred_file:
        pred_file.write('idx,pred,real\n')
        np.savetxt(pred_file, results, '%i,%3.7f,%3.7f')


def save_results(results, config):
    """ Create a file mapping mean and stddev to configurations

    Args:
        path (str): The path to which the file must be stored
        config (dict): A configuration given in a dict
        score (list): The accuracies of the configuration

    Example:
        >>> path = /path/to/sotre/
        >>> config = {'a':2, 'b':4}
        >>> save_results(path, [1, 40, 10, 15], config)
        >>> print(open(path + 'scores.txt').read())
        a           b       mean    std
        2           4       16.5    14.465476141489432
    """
    mstd_results = base.to_saveformat(results)
    _save_scores(results.scores, config, _FNAMES[_ACCURACIES])
    _save_scores(results.traintimes, config, _FNAMES[_TRAINTIMES])
    _save_scores(results.testtimes, config, _FNAMES[_TESTTIMES])


def _save_scores(scores, config, fname):
    path = SAVEDIR + fname + EXTENSION
    with open(path, 'a') as fscore:
        lines = _make_line(config.values(), scores)
        fscore.write(''.join(lines))


def _make_header(config_keys):
    fields = [key for key in config_keys]
    fields.extend(['mean', 'std'])
    return ','.join(fields) + '\n'


def _make_line(config, score=[]):
    line = [f"{conf}" for conf in config]
    line.extend([f"{value}" for value in score])
    return ','.join(line) + '\n'


def _make_conf_name(config):
    name = []
    for key in config:
        name.append(key)
        name.append(str(config[key]))
    return '_'.join(name)


def make_pred_name(dir_, config):
    """ Concatenate key_value to create a file name for a dict

    Args:
        dir_ (str): The path to the file
        config (dict): A configuration dictionary

    Returns:
        file_path (str): The full path + name of a configuration

    Example:
        >>> config = {'a':'3', 'b':4}
        >>> print(make_pred_name('home/user/path/', config))
        'home/user/path/a_2_b_4.preds'

    """
    fname = _make_conf_name(config)
    return f"{dir_}/{fname}{EXTENSION}"

