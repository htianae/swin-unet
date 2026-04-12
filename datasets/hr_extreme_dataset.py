import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class HRExtremeDataset(Dataset):
    DEFAULT_INPUT_KEYS = ("input", "inputs", "x", "image", "data")
    DEFAULT_TARGET_KEYS = ("target", "targets", "y", "label", "labels", "output")
    DEFAULT_MASK_KEYS = ("mask", "masks")

    def __init__(
        self,
        data_dir,
        input_keys=None,
        target_keys=None,
        return_mask=False,
        files=None,
        input_channels=None,
        target_channels=69,
    ):
        self.data_dir = data_dir
        self.return_mask = return_mask
        self.input_keys = tuple(input_keys or self.DEFAULT_INPUT_KEYS)
        self.target_keys = tuple(target_keys or self.DEFAULT_TARGET_KEYS)
        self.mask_keys = self.DEFAULT_MASK_KEYS
        self.input_channels = input_channels
        self.target_channels = target_channels
        if files is None:
            self.files = sorted(
                os.path.join(data_dir, file_name)
                for file_name in os.listdir(data_dir)
                if file_name.endswith(".npz")
            )
        else:
            self.files = list(files)
        if not self.files:
            raise ValueError("No .npz files found in {}".format(data_dir))

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _get_first_available_key(data, keys, kind):
        for key in keys:
            if key in data:
                return data[key]
        raise KeyError("Unable to find {} key in file. Tried keys: {}".format(kind, keys))

    @staticmethod
    def _reshape_channels(array, expected_channels, tensor_name):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim < 3:
            raise ValueError("{} must have at least 3 dimensions, got {}".format(tensor_name, array.shape))

        height, width = array.shape[-2:]
        array = array.reshape(-1, height, width)
        if expected_channels is not None and array.shape[0] != expected_channels:
            raise ValueError(
                "{} must have {} channels after reshape, got {}".format(
                    tensor_name, expected_channels, array.shape
                )
            )
        return array

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with np.load(file_path) as data:
            x = self._get_first_available_key(data, self.input_keys, "input")
            y = self._get_first_available_key(data, self.target_keys, "target")
            mask = None
            if self.return_mask:
                for key in self.mask_keys:
                    if key in data:
                        mask = data[key]
                        break

        x = self._reshape_channels(x, expected_channels=self.input_channels, tensor_name="input")
        y = self._reshape_channels(y, expected_channels=self.target_channels, tensor_name="target")

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.return_mask:
            if mask is None:
                mask = torch.ones((x.shape[-2], x.shape[-1]), dtype=torch.float32)
            else:
                mask = torch.tensor(np.asarray(mask, dtype=np.float32), dtype=torch.float32)
            return x, y, mask

        return x, y


def _validate_split_sizes(dataset_length, val_split, test_split):
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be in [0, 1), got {}".format(val_split))
    if not 0 <= test_split < 1:
        raise ValueError("test_split must be in [0, 1), got {}".format(test_split))
    if val_split + test_split >= 1:
        raise ValueError(
            "val_split + test_split must be < 1, got {} + {}".format(val_split, test_split)
        )
    if dataset_length < 3 and (val_split > 0 or test_split > 0):
        raise ValueError(
            "Need at least 3 samples to create train/val/test splits, got {}".format(dataset_length)
        )


def _make_dataset_from_files(root_dir, file_paths, input_channels, target_channels):
    return HRExtremeDataset(
        root_dir,
        files=file_paths,
        input_channels=input_channels,
        target_channels=target_channels,
    )


def _save_split_manifest(save_path, root_dir, seed, val_split, test_split, split_files):
    manifest = {
        "root_dir": os.path.abspath(root_dir),
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "splits": {
            split_name: [os.path.basename(path) for path in file_list]
            for split_name, file_list in split_files.items()
        },
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2, sort_keys=True)


def _load_split_manifest(split_file, root_dir):
    with open(split_file, "r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)

    manifest_root = os.path.abspath(manifest.get("root_dir", root_dir))
    root_dir = os.path.abspath(root_dir)
    if manifest_root != root_dir:
        raise ValueError(
            "Split file was created for root_dir={}, but current root_dir={}".format(
                manifest_root, root_dir
            )
        )

    split_files = {}
    for split_name, file_names in manifest["splits"].items():
        file_paths = [os.path.join(root_dir, file_name) for file_name in file_names]
        missing_files = [path for path in file_paths if not os.path.isfile(path)]
        if missing_files:
            raise FileNotFoundError(
                "Split file references missing files for '{}': {}".format(split_name, missing_files[:5])
            )
        split_files[split_name] = file_paths
    return split_files


def split_hr_extreme_dataset(dataset, seed, val_split=0.1, test_split=0.1):
    dataset_length = len(dataset)
    _validate_split_sizes(dataset_length, val_split, test_split)

    num_val = int(round(dataset_length * val_split))
    num_test = int(round(dataset_length * test_split))
    num_train = dataset_length - num_val - num_test

    if val_split > 0 and num_val == 0:
        num_val = 1
        num_train -= 1
    if test_split > 0 and num_test == 0:
        num_test = 1
        num_train -= 1

    if num_train <= 0:
        raise ValueError(
            "Split configuration leaves no training samples: total={}, val={}, test={}".format(
                dataset_length, num_val, num_test
            )
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_length).tolist()
    train_end = num_train
    val_end = num_train + num_val
    split_files = {
        "train": [dataset.files[index] for index in indices[:train_end]],
        "val": [dataset.files[index] for index in indices[train_end:val_end]],
        "test": [dataset.files[index] for index in indices[val_end:]],
    }
    return (
        _make_dataset_from_files(dataset.data_dir, split_files["train"]),
        _make_dataset_from_files(dataset.data_dir, split_files["val"]),
        _make_dataset_from_files(dataset.data_dir, split_files["test"]),
        split_files,
    )


def build_hr_extreme_datasets(
    root_dir,
    seed,
    val_split=0.1,
    test_split=0.1,
    split_file=None,
    save_split_file=None,
    input_channels=None,
    target_channels=69,
):
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
        return (
            HRExtremeDataset(train_dir, input_channels=input_channels, target_channels=target_channels),
            HRExtremeDataset(val_dir, input_channels=input_channels, target_channels=target_channels),
            HRExtremeDataset(test_dir, input_channels=input_channels, target_channels=target_channels),
        )

    if split_file and os.path.isfile(split_file):
        split_files = _load_split_manifest(split_file, root_dir)
        return (
            _make_dataset_from_files(root_dir, split_files["train"], input_channels, target_channels),
            _make_dataset_from_files(root_dir, split_files["val"], input_channels, target_channels),
            _make_dataset_from_files(root_dir, split_files["test"], input_channels, target_channels),
        )

    full_dataset = HRExtremeDataset(
        root_dir,
        input_channels=input_channels,
        target_channels=target_channels,
    )
    train_dataset, val_dataset, test_dataset, split_files = split_hr_extreme_dataset(
        full_dataset,
        seed=seed,
        val_split=val_split,
        test_split=test_split,
    )
    if save_split_file:
        _save_split_manifest(
            save_split_file,
            root_dir=root_dir,
            seed=seed,
            val_split=val_split,
            test_split=test_split,
            split_files=split_files,
        )
    return train_dataset, val_dataset, test_dataset
