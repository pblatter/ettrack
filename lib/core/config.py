import os
import yaml
from easydict import EasyDict as edict

config = edict()

# ------config for general parameters------
config.GPUS = "0,1,2,3"
config.WORKERS = 32
config.PRINT_FREQ = 10
config.OUTPUT_DIR = 'logs'
config.CHECKPOINT_DIR = 'snapshot'

config.ET = edict()
config.ET.TRAIN = edict()
config.ET.TEST = edict()
config.ET.REID = edict()
config.ET.TUNE = edict()
config.ET.DATASET = edict()
config.ET.DATASET.VID = edict()
config.ET.DATASET.GOT10K = edict()
config.ET.DATASET.COCO = edict()
config.ET.DATASET.DET = edict()
config.ET.DATASET.LASOT = edict()
config.ET.DATASET.YTB = edict()
config.ET.DATASET.VISDRONE = edict()
config.ET.DATASET.MIX = edict()

# own parameters
config.ET.DEVICE = 'cuda'

# augmentation
config.ET.DATASET.SHIFT = 4
config.ET.DATASET.SCALE = 0.05
config.ET.DATASET.COLOR = 1
config.ET.DATASET.FLIP = 0
config.ET.DATASET.BLUR = 0
config.ET.DATASET.GRAY = 0
config.ET.DATASET.MIXUP = 0
config.ET.DATASET.CUTOUT = 0
config.ET.DATASET.CHANNEL6 = 0
config.ET.DATASET.LABELSMOOTH = 0
config.ET.DATASET.ROTATION = 0
config.ET.DATASET.SHIFTs = 64
config.ET.DATASET.SCALEs = 0.18

config.ET.DATASET.MIX.DIST = 'beta'
config.ET.DATASET.MIX.ALPHA = 1.0
config.ET.DATASET.MIX.BETA = 1.0
config.ET.DATASET.MIX.MIN = 0
config.ET.DATASET.MIX.MAX = 1
config.ET.DATASET.MIX.PROB = 1

# vid
config.ET.DATASET.VID.PATH = '$data_path/vid/crop511'
config.ET.DATASET.VID.ANNOTATION = '$data_path/vid/train.json'

# got10k
config.ET.DATASET.GOT10K.PATH = '$data_path/got10k/crop511'
config.ET.DATASET.GOT10K.ANNOTATION = '$data_path/got10k/train.json'
config.ET.DATASET.GOT10K.RANGE = 100
config.ET.DATASET.GOT10K.USE = 200000

# visdrone
config.ET.DATASET.VISDRONE.ANNOTATION = '$data_path/visdrone/train.json'
config.ET.DATASET.VISDRONE.PATH = '$data_path/visdrone/crop271'
config.ET.DATASET.VISDRONE.RANGE = 100
config.ET.DATASET.VISDRONE.USE = 100000

# train
config.ET.TRAIN.SCRATCH = False
config.ET.TRAIN.EMA = 0.9998
config.ET.TRAIN.NEG_WEIGHT = 0.1
config.ET.TRAIN.GROUP = "resrchvc"
config.ET.TRAIN.EXID = "setting1"
config.ET.TRAIN.MODEL = "ET"
config.ET.TRAIN.RESUME = False
config.ET.TRAIN.START_EPOCH = 0
config.ET.TRAIN.END_EPOCH = 50
config.ET.TRAIN.TEMPLATE_SIZE = 127
config.ET.TRAIN.SEARCH_SIZE = 255
config.ET.TRAIN.STRIDE = 8
config.ET.TRAIN.BATCH = 32
config.ET.TRAIN.PRETRAIN = 'pretrain.model'
config.ET.TRAIN.LR_POLICY = 'log'
config.ET.TRAIN.LR = 0.001
config.ET.TRAIN.LR_END = 0.00001
config.ET.TRAIN.MOMENTUM = 0.9
config.ET.TRAIN.WEIGHT_DECAY = 0.0001
config.ET.TRAIN.WHICH_USE = ['GOT10K']  # VID or 'GOT10K'
config.ET.TRAIN.FREEZE_LAYER = []
# reid
config.ET.REID.PATCH_SIZE = 64
config.ET.REID.SAMPLE_PER_BATCH = 256
config.ET.REID.SAMPLE_PER_EPOCH = 4000
# test
config.ET.TEST.MODEL = config.ET.TRAIN.MODEL
config.ET.TEST.DATA = 'VOT2019'
config.ET.TEST.START_EPOCH = 30
config.ET.TEST.END_EPOCH = 50

# tune
config.ET.TUNE.MODEL = config.ET.TRAIN.MODEL
config.ET.TUNE.DATA = 'VOT2019'
config.ET.TUNE.METHOD = 'TPE'  # 'GENE' or 'RAY'



def _update_dict(k, v, model_name):
    if k in ['TRAIN', 'TEST', 'TUNE','REID']:
        for vk, vv in v.items():
            config[model_name][k][vk] = vv
    elif k == 'DATASET':
        for vk, vv in v.items():
            if vk not in ['VID', 'GOT10K', 'COCO', 'DET', 'YTB', 'LASOT']:
                config[model_name][k][vk] = vv
            else:
                for vvk, vvv in vv.items():
                    try:
                        config[model_name][k][vk][vvk] = vvv
                    except:
                        config[model_name][k][vk] = edict()
                        config[model_name][k][vk][vvk] = vvv

    else:
        config[k] = v   # gpu et.


def update_config(config_file):
    """
    ADD new keys to config
    """
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        model_name = list(exp_config.keys())[0]
        if model_name not in ['OCEAN', 'SIAMRPN', 'BASELINE', 'ET']:
            raise ValueError('please edit config.py to support new model')

        model_config = exp_config[model_name]  # siamfc or siamrpn
        for k, v in model_config.items():
            if k in config or k in config[model_name]:
                _update_dict(k, v, model_name)   # k=OCEAN or SIAMRPN
            else:
                raise ValueError("{} not exist in config.py".format(k))
