import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_debris_processed
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/debris_processed_33', help='root dir or npz file for data')
parser.add_argument('--dataset', type=str,
                    default='DebrisProcessed', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='', help='unused')
parser.add_argument('--num_classes', type=int,
                    default=33, help='output channel of network')
parser.add_argument('--history_steps', type=int,
                    default=2, help='number of input time steps')
parser.add_argument('--in_chans', type=int,
                    default=66, help='flattened input channels for config bookkeeping')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-4,
                    help='regression network learning rate')
parser.add_argument('--img_size', type=int,
                    default=320, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--val_path', type=str, default='',
                    help='deprecated for the single-directory baseline')
parser.add_argument('--val_split', type=float, default=0.1,
                    help='validation split used when train/val/test folders do not exist')
parser.add_argument('--test_split', type=float, default=0.1,
                    help='test split used when train/val/test folders do not exist')
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--eval_interval", default=1, type=int)
parser.add_argument("--log_interval", default=1, type=int,
                    help='print training loss every N batches')
parser.add_argument("--grad_clip", default=1.0, type=float,
                    help='gradient clipping norm for AdamW training')
parser.add_argument('--pretrained_ckpt', type=str, default=None,
                    help='optional pretrained checkpoint path; mismatched first-layer weights are skipped')
parser.add_argument('--target_step_index', type=int, default=0,
                    help='which future target step to use when targets contain multiple lead times')

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'DebrisProcessed': {
            'root_path': args.root_path,
            'list_dir': '',
            'num_classes': 33,
            'in_chans': args.in_chans,
            'history_steps': args.history_steps,
        },
    }
    if dataset_name not in dataset_config:
        raise ValueError("Unsupported dataset '{}'. This baseline is configured for DebrisProcessed.".format(dataset_name))

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.history_steps = dataset_config[dataset_name]['history_steps']
    args.in_chans = args.history_steps * args.num_classes
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    config = get_config(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)
    if config.MODEL.PRETRAIN_CKPT:
        net.load_from(config)

    trainer_debris_processed(args, net, args.output_dir)
