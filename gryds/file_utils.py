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
    fname = make_pred_name(path, config)
    results = np.vstack((predictions, sample_indexes, yreal)).T
    np.savetxt(fname, results, '%3.7f\t%4i\t%3.7f')


def save_scores(path, config, score):
    with open(path + '/scores.txt', 'a+') as fscore:
       conf_name = _make_conf_name(config)
       fscore.write(f"{conf_name}\t{score:3.7f}")


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
        >>> 'home/user/path/a_2_b_4.preds'

    """
    fname= _make_conf_name(config)
    return f"{dir_}/{fname}{PRED_EXTS}"


