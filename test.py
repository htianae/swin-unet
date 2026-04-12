import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.hr_extreme_dataset import build_hr_extreme_datasets
from networks.vision_transformer import SwinUnet as ViT_seg


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/HR-Extreme',
                    help='root dir containing either split folders or a flat directory of .npz files')
parser.add_argument('--dataset', type=str,
                    default='HRExtreme', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=69, help='output channel of network')
parser.add_argument('--in_chans', type=int,
                    default=138, help='input channel of network')
parser.add_argument('--output_dir', type=str, required=True, help='directory containing checkpoints')
parser.add_argument('--batch_size', type=int, default=4, help='batch size per gpu')
parser.add_argument('--img_size', type=int, default=160, help='input patch size of network input')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic inference')
parser.add_argument('--base_lr', type=float, default=0.01, help='unused placeholder to preserve config API')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs.",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, full: cache all data, part: cache one shard')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                    help='dataset split to evaluate')
parser.add_argument('--val_split', type=float, default=0.1,
                    help='validation split used when train/val/test folders do not exist')
parser.add_argument('--test_split', type=float, default=0.1,
                    help='test split used when train/val/test folders do not exist')
parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers')
parser.add_argument('--checkpoint', type=str, default='',
                    help='optional checkpoint path; defaults to <output_dir>/best_model.pth')
parser.add_argument('--split_file', type=str, default='',
                    help='optional dataset split manifest; defaults to <output_dir>/dataset_split.json')
parser.add_argument('--save_predictions', action='store_true',
                    help='save prediction and target tensors to output_dir/predictions as .npz files')

args = parser.parse_args()


def _compute_rmse(prediction, target):
    return torch.sqrt(((prediction - target) ** 2).mean())


def _select_dataset_split(args):
    split_file = args.split_file or os.path.join(args.output_dir, 'dataset_split.json')
    train_dataset, val_dataset, test_dataset = build_hr_extreme_datasets(
        args.root_path,
        seed=args.seed,
        val_split=args.val_split,
        test_split=args.test_split,
        split_file=split_file if os.path.isfile(split_file) else None,
        input_channels=args.in_chans,
        target_channels=args.num_classes,
    )
    split_map = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
    }
    selected_dataset = split_map[args.split]
    if len(selected_dataset) == 0:
        raise ValueError("Selected '{}' split is empty.".format(args.split))
    return selected_dataset


def _resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    return os.path.join(args.output_dir, 'best_model.pth')


def inference(args, model, dataset, device, save_dir=None):
    testloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info("%d %s iterations", len(testloader), args.split)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_rmse = 0.0

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), desc=args.split):
            image_batch, label_batch = sampled_batch
            image_batch = image_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            label_batch = label_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            pred_batch = model(image_batch)
            loss = criterion(pred_batch, label_batch)
            rmse = _compute_rmse(pred_batch, label_batch)

            total_loss += loss.item()
            total_rmse += rmse.item()

            if save_dir is not None:
                for sample_offset in range(pred_batch.shape[0]):
                    sample_id = i_batch * args.batch_size + sample_offset
                    save_path = os.path.join(save_dir, "sample_{:05d}.npz".format(sample_id))
                    np.savez_compressed(
                        save_path,
                        prediction=pred_batch[sample_offset].cpu().numpy().astype(np.float32),
                        target=label_batch[sample_offset].cpu().numpy().astype(np.float32),
                    )

    mean_loss = total_loss / len(testloader)
    mean_rmse = total_rmse / len(testloader)
    logging.info('%s loss: %f, rmse: %f', args.split, mean_loss, mean_rmse)
    return mean_loss, mean_rmse


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

    dataset_config = {
        'HRExtreme': {
            'root_path': args.root_path,
            'num_classes': 69,
            'in_chans': args.in_chans,
        },
    }
    if args.dataset not in dataset_config:
        raise ValueError("Unsupported dataset '{}'. This baseline is configured for HRExtreme.".format(args.dataset))

    args.num_classes = dataset_config[args.dataset]['num_classes']
    args.in_chans = dataset_config[args.dataset]['in_chans']
    args.root_path = dataset_config[args.dataset]['root_path']
    config = get_config(args)

    checkpoint_path = _resolve_checkpoint(args)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    dataset = _select_dataset_split(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    msg = net.load_state_dict(state_dict, strict=False)

    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, "{}_{}.txt".format(args.split, os.path.basename(checkpoint_path)))
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("checkpoint load result: %s", msg)
    logging.info("evaluating %s split with %d samples", args.split, len(dataset))
    logging.info("split manifest: %s", args.split_file or os.path.join(args.output_dir, 'dataset_split.json'))

    save_dir = None
    if args.save_predictions:
        save_dir = os.path.join(args.output_dir, 'predictions', args.split)
        os.makedirs(save_dir, exist_ok=True)

    inference(args, net, dataset, device=device, save_dir=save_dir)
