import os
import sys

import numpy as np


PRED_EXTS = '.preds'
SCORE_EXTS = '.scores'


def find_files(path, extension):
    """ Gather the name of files of a specific extension under a directory

    Args:
        path (str): The absolute path of the directory
        extension (str): The extension to lookup

    Returns:
        files (list): The list of absolute path for each file
    """
    files = []
    extension = '.' + extension.lower()
    for base, directories, fnames in os.walk(path):
        for fname in fnames:
            if fname.lower().endswith(extension):
                files.append(os.path.join(base, fname))
    return files


def find_txt_files(path):
    """ Find every .txt file under a directory

    Note:
        Uses find_files method with txt extension
    """
    return find_files(path, 'txt')


def find_wav_files(path):
    """ Find every .wav file under a directory

    Note:
        Uses find_files method with wav extension
    """
    return find_files(path, 'wav')


def save_predictions(path, config, predictions, sample_indexes, yreal):
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
    fname = make_pred_name(path, config)
    results = np.vstack((predictions, sample_indexes, yreal)).T
    with open(fname, 'a+') as pred_file:
        np.savetxt(pred_file, results, '%3.7f\t%4i\t%3.7f')


def save_scores(path, config, scores):
    """ Create a file mapping mean and std dev to configurations

    Args:
        path (str): The path to which the file must be stored
        config (dict): A configuration given in a dict
        score (list): The accuracies of the configuration

    Example:
        >>> path = /path/to/sotre/
        >>> config = {'a':2, 'b':4}
        >>> save_scores(path, config, [1, 40, 10, 15])
        >>> print(open(path + 'scores.txt').read())
        a_2_b_4     16.5    14.465476141489432
    """
    with open(path + '/scores.txt', 'a+') as fscore:
       conf_name = _make_conf_name(config)
       mean = np.mean(scores)
       std = np.std(scores)
       fscore.write(f"{conf_name}\t{mean:3.7f}\t{std:3.7f}\n")


def _make_conf_name(config):
    conf_name = []
    for key in config.keys():
        conf_name.append(key)
        conf_name.append(str(config[key]))
    return '_'.join(conf_name)


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
    return f"{dir_}/{fname}{PRED_EXTS}"


