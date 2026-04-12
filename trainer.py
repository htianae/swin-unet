from functools import partial
import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.hr_extreme_dataset import build_hr_extreme_datasets


def _seed_worker(worker_id, base_seed):
    random.seed(base_seed + worker_id)


def _build_hr_extreme_datasets(args, split_file):
    train_dataset, val_dataset, test_dataset = build_hr_extreme_datasets(
        args.root_path,
        seed=args.seed,
        val_split=args.val_split,
        test_split=args.test_split,
        split_file=split_file,
        save_split_file=split_file,
        input_channels=args.in_chans,
        target_channels=args.num_classes,
        target_step_index=args.target_step_index,
    )
    logging.info(
        "Dataset split sizes -> train: %d, val: %d, test: %d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    return train_dataset, val_dataset


def _compute_rmse(prediction, target):
    return torch.sqrt(((prediction - target) ** 2).mean())


def trainer_hr_extreme(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    device = next(model.parameters()).device
    pin_memory = device.type == "cuda"
    worker_init_fn = partial(_seed_worker, base_seed=args.seed)
    split_file = os.path.join(snapshot_path, "dataset_split.json")

    db_train, db_val = _build_hr_extreme_datasets(args, split_file=split_file)
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))
    logging.info("Dataset split manifest: %s", split_file)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin_memory,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin_memory,
                            worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = float("inf")
    for epoch_num in iterator:
        model.train()
        running_train_loss = 0.0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch
            image_batch = image_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            label_batch = label_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            outputs = model(image_batch)
            loss = criterion(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('train/loss_iter', loss.item(), iter_num)
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        logging.info('Train epoch: %d : loss : %f' % (epoch_num, epoch_train_loss))
        print("Epoch {}/{} Train Loss: {:.6f}".format(epoch_num + 1, max_epoch, epoch_train_loss), flush=True)
        writer.add_scalar('train/loss_epoch', epoch_train_loss, epoch_num)
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            running_val_loss = 0.0
            running_val_rmse = 0.0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch
                    image_batch = image_batch.to(device=device, dtype=torch.float32, non_blocking=True)
                    label_batch = label_batch.to(device=device, dtype=torch.float32, non_blocking=True)
                    outputs = model(image_batch)
                    loss = criterion(outputs, label_batch)
                    rmse = _compute_rmse(outputs, label_batch)
                    running_val_loss += loss.item()
                    running_val_rmse += rmse.item()

                batch_loss = running_val_loss / len(val_loader)
                batch_rmse = running_val_rmse / len(val_loader)
                writer.add_scalar('val/loss', batch_loss, epoch_num)
                writer.add_scalar('val/rmse', batch_rmse, epoch_num)
                logging.info('Val epoch: %d : loss : %f, rmse: %f' % (
                    epoch_num, batch_loss, batch_rmse))
                print(
                    "Epoch {}/{} Val Loss: {:.6f}, RMSE: {:.6f}".format(
                        epoch_num + 1, max_epoch, batch_loss, batch_rmse
                    ),
                    flush=True,
                )
                if batch_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = batch_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"
