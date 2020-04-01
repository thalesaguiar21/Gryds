import os
import sys

import numpy as np

from . import confs


PRED_EXTS = '.preds'
SAVEDIR = confs.get_savedir()

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
        directories.sort()
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
    results = np.vstack((predictions, sample_indexes, yreal)).T
    with open(fname, 'a+') as pred_file:
        np.savetxt(pred_file, results, '%3.7f\t%4i\t%3.7f')


def save_times(times, config):
    with open(SAVEDIR + '/times.txt', 'w+') as ftime:
        header = _make_header(config.keys())
        lines = [_make_line(t) for t in times]
        ftime.write(header)
        ftime.write(''.join(lines))


def save_scores(scores, config):
    """ Create a file mapping mean and std dev to configurations

    Args:
        path (str): The path to which the file must be stored
        config (dict): A configuration given in a dict
        score (list): The accuracies of the configuration

    Example:
        >>> path = /path/to/sotre/
        >>> config = {'a':2, 'b':4}
        >>> save_scores(path, [1, 40, 10, 15], config)
        >>> print(open(path + 'scores.txt').read())
        a           b       mean    std
        2           4       16.5    14.465476141489432
    """
    with open(SAVEDIR + '/scores.txt', 'w+') as fscore:
        header = _make_header(config.keys())
        lines = [_make_line(score) for score in scores]
        fscore.write(header)
        fscore.write(''.join(lines))


def _make_header(config_keys):
    fields = [key for key in config_keys]
    fields.extend(['mean', 'std'])
    return _make_line(fields)


def _make_line(score):
    line = [f"{value:<17}\t" for value in score]
    return ''.join(line) + '\n'


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
    return f"{dir_}/{fname}{PRED_EXTS}"

