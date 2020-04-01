import os

paths = {
    'save': os.path.abspath('tests') + '/'
}

extensions = {
    'all': '.gs'
}

metrics = {
    'timeunit': 'nano'
}

_timeunit = {
    'sec': 1,
    'mili': 1e-3,
    'nano': 1e-9,
}

def get_savedir():
    return paths['save']

def get_extension():
    return extensions['all']

def get_timeunit():
    return _timeunit.get(metrics['timeunit'], 1)

