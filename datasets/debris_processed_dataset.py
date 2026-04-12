import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class DebrisProcessedDataset(Dataset):
    DEFAULT_INPUT_KEYS = ("X", "inputs")
    DEFAULT_TARGET_KEYS = ("y", "targets")
    DEFAULT_MASK_KEYS = ("masks", "mask")

    def __init__(
        self,
        data_path,
        input_keys=None,
        target_keys=None,
        return_mask=True,
        sample_refs=None,
        history_steps=2,
        vars_per_step=33,
        target_step_index=0,
    ):
        self.data_path = os.path.abspath(data_path)
        self.root_dir = self.data_path if os.path.isdir(self.data_path) else os.path.dirname(self.data_path)
        self.return_mask = return_mask
        self.input_keys = tuple(input_keys or self.DEFAULT_INPUT_KEYS)
        self.target_keys = tuple(target_keys or self.DEFAULT_TARGET_KEYS)
        self.mask_keys = tuple(self.DEFAULT_MASK_KEYS)
        self.history_steps = history_steps
        self.vars_per_step = vars_per_step
        self.target_step_index = target_step_index

        if sample_refs is None:
            self.files = self._discover_files(self.data_path)
            self.sample_refs = self._build_sample_refs(self.files)
        else:
            self.sample_refs = list(sample_refs)
            self.files = sorted({file_path for file_path, _ in self.sample_refs})

        if not self.sample_refs:
            raise ValueError("No samples found in {}".format(self.data_path))

    def __len__(self):
        return len(self.sample_refs)

    @staticmethod
    def _discover_files(data_path):
        if os.path.isfile(data_path) and data_path.endswith(".npz"):
            return [data_path]
        if os.path.isdir(data_path):
            files = sorted(
                os.path.join(data_path, file_name)
                for file_name in os.listdir(data_path)
                if file_name.endswith(".npz")
            )
            if files:
                return files
        raise ValueError("No .npz files found in {}".format(data_path))

    def _build_sample_refs(self, files):
        sample_refs = []
        for file_path in files:
            with np.load(file_path) as data:
                input_array = self._get_first_available_key(data, self.input_keys, "input")
                num_samples = int(input_array.shape[0])
            sample_refs.extend((file_path, sample_index) for sample_index in range(num_samples))
        return sample_refs

    @staticmethod
    def _get_first_available_key(data, keys, kind):
        for key in keys:
            if key in data:
                return data[key]
        raise KeyError("Unable to find {} key in file. Tried keys: {}".format(kind, keys))

    def _reshape_input(self, array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError("input sample must have 4 dimensions, got {}".format(array.shape))
        if array.shape[0] != self.history_steps:
            raise ValueError(
                "input sample must have {} history steps, got {}".format(self.history_steps, array.shape)
            )
        if array.shape[-1] != self.vars_per_step:
            raise ValueError(
                "input sample must have {} variables in last dim, got {}".format(
                    self.vars_per_step, array.shape
                )
            )
        return np.transpose(array, (0, 3, 1, 2))

    def _reshape_target(self, array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError("target sample must have 4 dimensions, got {}".format(array.shape))
        if array.shape[-1] != self.vars_per_step:
            raise ValueError(
                "target sample must have {} variables in last dim, got {}".format(
                    self.vars_per_step, array.shape
                )
            )
        if not 0 <= self.target_step_index < array.shape[0]:
            raise ValueError(
                "target_step_index {} is out of range for target shape {}".format(
                    self.target_step_index, array.shape
                )
            )
        array = array[self.target_step_index]
        return np.transpose(array, (2, 0, 1))

    @staticmethod
    def _reshape_mask(array, height, width):
        array = np.asarray(array, dtype=np.float32)
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError("mask must resolve to shape (H, W), got {}".format(array.shape))
        if array.shape != (height, width):
            raise ValueError("mask shape {} does not match target shape ({}, {})".format(array.shape, height, width))
        return array

    def __getitem__(self, idx):
        file_path, sample_index = self.sample_refs[idx]
        with np.load(file_path) as data:
            input_array = self._get_first_available_key(data, self.input_keys, "input")
            target_array = self._get_first_available_key(data, self.target_keys, "target")
            x = input_array[sample_index]
            y = target_array[sample_index]
            mask = None
            if self.return_mask:
                for key in self.mask_keys:
                    if key in data:
                        mask_data = data[key]
                        if mask_data.shape[0] == input_array.shape[0]:
                            mask = mask_data[sample_index]
                        else:
                            mask = mask_data
                        break

        x = self._reshape_input(x)
        y = self._reshape_target(y)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.return_mask:
            if mask is None:
                mask = torch.ones((y.shape[-2], y.shape[-1]), dtype=torch.float32)
            else:
                mask = self._reshape_mask(mask, y.shape[-2], y.shape[-1])
                mask = torch.tensor(mask, dtype=torch.float32)
            return x, y, mask

        return x, y


def _normalize_root_dir(data_path):
    data_path = os.path.abspath(data_path)
    return data_path if os.path.isdir(data_path) else os.path.dirname(data_path)


def _validate_split_sizes(dataset_length, val_split, test_split):
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be in [0, 1), got {}".format(val_split))
    if not 0 <= test_split < 1:
        raise ValueError("test_split must be in [0, 1), got {}".format(test_split))
    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split must be < 1, got {} + {}".format(val_split, test_split))
    if dataset_length < 3 and (val_split > 0 or test_split > 0):
        raise ValueError("Need at least 3 samples to create train/val/test splits, got {}".format(dataset_length))


def _make_dataset_from_refs(data_path, sample_refs, history_steps, vars_per_step, target_step_index):
    return DebrisProcessedDataset(
        data_path,
        sample_refs=sample_refs,
        return_mask=True,
        history_steps=history_steps,
        vars_per_step=vars_per_step,
        target_step_index=target_step_index,
    )


def _save_split_manifest(save_path, root_dir, seed, val_split, test_split, split_refs):
    manifest = {
        "root_dir": os.path.abspath(root_dir),
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "splits": {
            split_name: [
                {
                    "file": os.path.relpath(file_path, root_dir),
                    "sample_index": sample_index,
                }
                for file_path, sample_index in refs
            ]
            for split_name, refs in split_refs.items()
        },
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2, sort_keys=True)


def _load_split_manifest(split_file, data_path):
    with open(split_file, "r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)

    root_dir = _normalize_root_dir(data_path)
    manifest_root = os.path.abspath(manifest.get("root_dir", root_dir))
    if manifest_root != root_dir:
        raise ValueError(
            "Split file was created for root_dir={}, but current root_dir={}".format(
                manifest_root, root_dir
            )
        )

    split_refs = {}
    for split_name, refs in manifest["splits"].items():
        reconstructed_refs = []
        for ref in refs:
            file_path = os.path.join(root_dir, ref["file"])
            if not os.path.isfile(file_path):
                raise FileNotFoundError("Split file references missing file {}".format(file_path))
            reconstructed_refs.append((file_path, int(ref["sample_index"])))
        split_refs[split_name] = reconstructed_refs
    return split_refs


def split_debris_processed_dataset(dataset, seed, val_split=0.1, test_split=0.1):
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
    split_refs = {
        "train": [dataset.sample_refs[index] for index in indices[:train_end]],
        "val": [dataset.sample_refs[index] for index in indices[train_end:val_end]],
        "test": [dataset.sample_refs[index] for index in indices[val_end:]],
    }
    return (
        _make_dataset_from_refs(
            dataset.data_path,
            split_refs["train"],
            dataset.history_steps,
            dataset.vars_per_step,
            dataset.target_step_index,
        ),
        _make_dataset_from_refs(
            dataset.data_path,
            split_refs["val"],
            dataset.history_steps,
            dataset.vars_per_step,
            dataset.target_step_index,
        ),
        _make_dataset_from_refs(
            dataset.data_path,
            split_refs["test"],
            dataset.history_steps,
            dataset.vars_per_step,
            dataset.target_step_index,
        ),
        split_refs,
    )


def build_debris_processed_datasets(
    data_path,
    seed,
    val_split=0.1,
    test_split=0.1,
    split_file=None,
    save_split_file=None,
    history_steps=2,
    vars_per_step=33,
    target_step_index=0,
):
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    test_dir = os.path.join(data_path, "test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
        return (
            DebrisProcessedDataset(train_dir, return_mask=True, history_steps=history_steps,
                                   vars_per_step=vars_per_step, target_step_index=target_step_index),
            DebrisProcessedDataset(val_dir, return_mask=True, history_steps=history_steps,
                                   vars_per_step=vars_per_step, target_step_index=target_step_index),
            DebrisProcessedDataset(test_dir, return_mask=True, history_steps=history_steps,
                                   vars_per_step=vars_per_step, target_step_index=target_step_index),
        )

    if split_file and os.path.isfile(split_file):
        split_refs = _load_split_manifest(split_file, data_path)
        return (
            _make_dataset_from_refs(data_path, split_refs["train"], history_steps, vars_per_step, target_step_index),
            _make_dataset_from_refs(data_path, split_refs["val"], history_steps, vars_per_step, target_step_index),
            _make_dataset_from_refs(data_path, split_refs["test"], history_steps, vars_per_step, target_step_index),
        )

    full_dataset = DebrisProcessedDataset(
        data_path,
        return_mask=True,
        history_steps=history_steps,
        vars_per_step=vars_per_step,
        target_step_index=target_step_index,
    )
    train_dataset, val_dataset, test_dataset, split_refs = split_debris_processed_dataset(
        full_dataset,
        seed=seed,
        val_split=val_split,
        test_split=test_split,
    )
    if save_split_file:
        _save_split_manifest(
            save_split_file,
            root_dir=_normalize_root_dir(data_path),
            seed=seed,
            val_split=val_split,
            test_split=test_split,
            split_refs=split_refs,
        )
    return train_dataset, val_dataset, test_dataset
