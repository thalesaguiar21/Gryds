import os
from .context import gryds
from gryds import confs

confs.paths['save'] = os.path.abspath('tests/') + '/'

