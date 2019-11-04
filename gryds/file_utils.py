import os
import sys


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


def save_predictions(path, predictions, sample_indexes, yreal):
    results = np.hstack((predictions, sample_indexes, yreal))
    np.savetxt(path, results, '%3.7f\t%4i\t%3.7f')

