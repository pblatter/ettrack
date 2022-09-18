# import supernet_backbone.tools._init_paths
import os
import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime
from copy import deepcopy

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False


from supernet_backbone.lib_back.models.hypernet import _gen_supernet

from supernet_backbone.lib_back.utils.helpers import *


def build_supernet(flops_maximum=600):

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, sta_num, size_factor = _gen_supernet(
        flops_minimum= 0,
        flops_maximum= flops_maximum,
        num_classes= 1000,
        drop_rate= 0.0,
        global_pool='avg',
        resunit=False,
        dil_conv=False,
        slice=4)

    return model, sta_num

#  Backbone with Dynamic output position
def build_supernet_DP(flops_maximum=600):

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    '''2020.10.14 Set DP = True'''
    model, sta_num, size_factor = _gen_supernet(
        flops_minimum= 0,
        flops_maximum= flops_maximum,
        DP=True,
        num_classes= 1000,
        drop_rate= 0.0,
        global_pool='avg',
        resunit=False,
        dil_conv=False,
        slice=4)

    return model, sta_num

if __name__ == '__main__':
    _, sta_num = build_supernet(flops_maximum=600)
    print(sta_num)
