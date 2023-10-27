from pathlib import Path

import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10

from cifar10_ssl.transforms import tf_preproc


def get_dataloaders(
        dataset_dir: Path | str,
        train_lbl_size: float | int,
        train_unl_size: float | int,
        test_size: float | int,
        batch_size: int = 48,
        num_workers: int = 0,
        seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns tuple of train (labelled, unlabeled), and test dataloaders
     as three torch.utils.data.DataLoader objects.
     """

    src_train_ds = CIFAR10(
        dataset_dir,
        train=True,
        download=True,
        transform=tf_preproc,
    )
    src_test_ds = CIFAR10(
        dataset_dir,
        train=False,
        download=True,
        transform=tf_preproc,
    )
    classes = src_train_ds.classes
    ohe = OneHotEncoder().fit([classes])
    n_classes = len(classes)

    train_lbl_ds, train_unl_ds, test_ds, _ = random_split(
        src_train_ds,
        [train_lbl_size, train_unl_size, test_size,
         1 - train_lbl_size - train_unl_size - test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # We use drop_last=True to ensure that the batch size is always the same
    # This is crucial as we need to average the predictions across the batch
    # size axis.
    train_lbl_dl = DataLoader(
        train_lbl_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    train_unl_dl = DataLoader(
        train_unl_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_lbl_dl, train_unl_dl, test_dl
