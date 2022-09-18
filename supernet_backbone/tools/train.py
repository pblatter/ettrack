import warnings
warnings.filterwarnings('ignore')
import _init_paths
import argparse
import yaml
import torch.nn as nn
import numpy as np

from torchscope import scope
from datetime import datetime

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False

from lib.dataset import Dataset, create_loader, resolve_data_config
from lib.core.train_function import train_epoch
from lib.core.valid_function import validate
from lib.models.model import _gen_childnet
from lib.utils.helpers import *
from lib.utils.EMA import ModelEma
from lib.utils.saver import CheckpointSaver
from lib.utils.loss import LabelSmoothingCrossEntropy
from lib.utils.optimizer import create_optimizer
from lib.utils.scheduler import create_scheduler
from torch.utils.tensorboard import SummaryWriter

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='./lib/configs/14.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('--name', default='train', type=str)
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=4, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 4)')
parser.add_argument('--pool-bn', action='store_true', default=False,
                    help='Use batch norm after classifier')
parser.add_argument('--zero-gamma', action='store_true', default=False,
                    help='use zero gamma initialization')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=0.0, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=0.0, metavar='PCT',
                    help='Drop block rate (default: None)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),

parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')
# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')
# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logger training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='prec1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "prec1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--accumulation-steps", default=1, type=int)
parser.add_argument('--loss-scale', action='store_true',
                    help='add loss scale.')
parser.add_argument('--maxup', action='store_true')
parser.add_argument('--model_selection', default=14, type=int)

local_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)
word_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)
dist_url = 'tcp://' + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT']

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()

    if args.local_rank == 0:
        args.local_rank = local_rank
        print("rank:{0},word_size:{1},dist_url:{2}".format(local_rank, word_size, dist_url))

    if args.model_selection == 470:
        arch_list = [[0], [3, 4, 3, 1], [3, 2, 3, 0], [3, 3, 3, 1], [3, 3, 3, 3], [3, 3, 3, 3], [0]]
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            # stage 1, 112x112 in
            ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25',
             'ir_r1_k3_s1_e4_c24_se0.25'],
            # stage 2, 56x56 in
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25',
             'ir_r1_k5_s2_e4_c40_se0.25'],
            # stage 3, 28x28 in
            ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25',
             'ir_r2_k3_s1_e4_c80_se0.25'],
            # stage 4, 14x14in
            ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
             'ir_r1_k3_s1_e6_c96_se0.25'],
            # stage 5, 14x14in
            ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25',
             'ir_r1_k5_s2_e6_c192_se0.25'],
            # stage 6, 7x7 in
            ['cn_r1_k1_s1_c320_se0.25'],
        ]
        args.img_size = 224
    elif args.model_selection == 42:
        arch_list = [[0], [3], [3, 1], [3, 1], [3, 3, 3], [3, 3], [0]]
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            # stage 1, 112x112 in
            ['ir_r1_k3_s2_e4_c24_se0.25'],
            # stage 2, 56x56 in
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'],
            # stage 3, 28x28 in
            ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s2_e6_c80_se0.25'],
            # stage 4, 14x14in
            ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25'],
            # stage 5, 14x14in
            ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'],
            # stage 6, 7x7 in
            ['cn_r1_k1_s1_c320_se0.25'],
        ]
        args.img_size = 96
    elif args.model_selection == 14:
        arch_list = [[0], [3], [3, 3], [3, 3], [3], [3], [0]]
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            # stage 1, 112x112 in
            ['ir_r1_k3_s2_e4_c24_se0.25'],
            # stage 2, 56x56 in
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k3_s2_e4_c40_se0.25'],
            # stage 3, 28x28 in
            ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s2_e4_c80_se0.25'],
            # stage 4, 14x14in
            ['ir_r1_k3_s1_e6_c96_se0.25'],
            # stage 5, 14x14in
            ['ir_r1_k5_s2_e6_c192_se0.25'],
            # stage 6, 7x7 in
            ['cn_r1_k1_s1_c320_se0.25'],
        ]
        args.img_size = 64
    elif args.model_selection == 112:
        arch_list = [[0], [3], [3, 3], [3, 3], [3, 3, 3], [3, 3], [0]]
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            # stage 1, 112x112 in
            ['ir_r1_k3_s2_e4_c24_se0.25'],
            # stage 2, 56x56 in
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k3_s2_e4_c40_se0.25'],
            # stage 3, 28x28 in
            ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s2_e6_c80_se0.25'],
            # stage 4, 14x14in
            ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25'],
            # stage 5, 14x14in
            ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'],
            # stage 6, 7x7 in
            ['cn_r1_k1_s1_c320_se0.25'],
        ]
        args.img_size = 160
    elif args.model_selection == 285:
        arch_list = [[0], [3], [3, 3], [3, 1, 3], [3, 3, 3, 3], [3, 3, 3], [0]]
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            # stage 1, 112x112 in
            ['ir_r1_k3_s2_e4_c24_se0.25'],
            # stage 2, 56x56 in
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'],
            # stage 3, 28x28 in
            ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s2_e6_c80_se0.25'],
            # stage 4, 14x14in
            ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
             'ir_r1_k3_s1_e6_c96_se0.25'],
            # stage 5, 14x14in
            ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'],
            # stage 6, 7x7 in
            ['cn_r1_k1_s1_c320_se0.25'],
        ]
        args.img_size = 224
    elif args.model_selection == 600:
        arch_list = [[0], [3, 3, 2, 3, 3], [3, 2, 3, 2, 3], [3, 2, 3, 2, 3], [3, 3, 2, 2, 3, 3], [3, 3, 2, 3, 3, 3],
                     [0]]
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            # stage 1, 112x112 in
            ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s2_e4_c24_se0.25',
             'ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s2_e4_c24_se0.25'],
            # stage 2, 56x56 in
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25',
             'ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'],
            # stage 3, 28x28 in
            ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25',
             'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25'],
            # stage 4, 14x14in
            ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
             'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25'],
            # stage 5, 14x14in
            ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25',
             'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25'],
            # stage 6, 7x7 in
            ['cn_r1_k1_s1_c320_se0.25'],
        ]
        args.img_size = 224
    else:
        raise ValueError("Not Supported!")

    model = _gen_childnet(
        arch_list,
        arch_def,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        pool_bn=args.pool_bn,
        zero_gamma=args.zero_gamma)

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './experiments'
        exp_name = '-'.join([
            args.name,
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            str(data_config['input_size'][-1])
        ])
        output_dir = get_outdir(output_base, 'retrain', exp_name)
        logger = get_logger(os.path.join(output_dir, 'retrain.log'))
        writer = SummaryWriter(os.path.join(output_dir, 'runs'))
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            f.write(args_text)
    else:
        writer = None
        logger = None

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1 and args.local_rank == 0:
            logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed. Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    args.distributed = True
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.local_rank == 0:
        if args.distributed:
            logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                        % (args.rank, args.world_size))
        else:
            logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.local_rank == 0:
        scope(model, input_size=(3, 224, 224))

    if os.path.exists(args.initial_checkpoint):
        load_checkpoint(model, args.initial_checkpoint)

    if args.local_rank == 0:
        logger.info('Model %s created, param count: %d' %
                    (args.model, sum([m.numel() for m in model.parameters()])))

    if args.num_gpu > 1:
        if args.amp:
            if args.local_rank == 0:
                logger.warning(
                    'AMP does not work well with nn.DataParallel, disabling. Use distributed mode for multi-GPU AMP.')
            args.amp = False
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()

    optimizer = create_optimizer(args, model)

    use_amp = False
    if has_apex and args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        use_amp = True
    if args.local_rank == 0:
        logger.info('NVIDIA APEX {}. '
                    ' {}.'.format(
            'installed' if has_apex else 'not installed', 'on' if use_amp else 'off'))

    # optionally resume from a checkpoint
    resume_state = {}
    resume_epoch = None
    if args.resume:
        resume_state, resume_epoch = resume_checkpoint(model, args.resume)
    if resume_state and not args.no_resume_opt:
        if 'optimizer' in resume_state:
            if args.local_rank == 0:
                logging.info('Restoring Optimizer state from checkpoint')
            optimizer.load_state_dict(resume_state['optimizer'])
        if use_amp and 'amp' in resume_state and 'load_state_dict' in amp.__dict__:
            if args.local_rank == 0:
                logging.info('Restoring NVIDIA AMP state from checkpoint')
            amp.load_state_dict(resume_state['amp'])
    del resume_state

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex:
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    logger.info('Converted model to use Synchronized BatchNorm.')
            except Exception as e:
                if args.local_rank == 0:
                    logger.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex:
            model = DDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logger.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            model = DDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP


    train_dir = os.path.join(args.data, 'train')
    if not os.path.exists(train_dir) and args.local_rank == 0:
        logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    eval_dir = os.path.join(args.data, 'val')
    if not os.path.exists(eval_dir) and args.local_rank == 0:
        logger.error('Validation folder does not exist at: {}'.format(eval_dir))
        exit(1)
    dataset_eval = Dataset(eval_dir)

    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=0,
        interpolation=args.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=None,
        pin_memory=args.pin_mem,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    if args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        validate_loss_fn = train_loss_fn

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logger.info('Scheduled epochs: {}'.format(num_epochs))

    try:
        best_record = 0
        best_ep = 0
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                use_amp=use_amp, model_ema=model_ema, logger=logger, writer=writer)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    logging.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(epoch, model, loader_eval, validate_loss_fn, args, logger=logger, writer=writer)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                ema_eval_metrics = validate(epoch,
                                            model_ema.ema, loader_eval, validate_loss_fn, args, log_suffix=' (EMA)',
                                            logger=logger, writer=writer)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, args,
                    epoch=epoch, model_ema=model_ema, metric=save_metric, use_amp=use_amp)

            if best_record < eval_metrics[eval_metric]:
                best_record = eval_metrics[eval_metric]
                best_ep = epoch

            if args.local_rank == 0:
                logger.info('*** Best metric: {0} (epoch {1})'.format(best_record, best_ep))

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

if __name__ == '__main__':
    main()
