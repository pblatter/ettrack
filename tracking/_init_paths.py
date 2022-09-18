from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

project_path = osp.join(this_dir, '..')
add_path(project_path)

#pytracking_path = osp.join(this_dir, '..', 'pytracking')
#add_path(pytracking_path)

print(f'sys path:')
for i in sys.path:
    print(f'{i}')