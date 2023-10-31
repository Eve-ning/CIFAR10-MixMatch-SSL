from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from cifar10_ssl.transforms import tf_preproc


def get_dataloaders(
    dataset_dir: Path | str,
    train_lbl_size: float = 0.005,
    train_unl_size: float = 0.980,
    batch_size: int = 48,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, list[str]]:
    """Get the dataloaders for the CIFAR10 dataset.

    Notes:
        The train_lbl_size and train_unl_size must sum to less than 1.
        The leftover data is used for the validation set.

    Args:
        dataset_dir: The directory where the dataset is stored.
        train_lbl_size: The size of the labelled training set.
        train_unl_size: The size of the unlabelled training set.
        batch_size: The batch size.
        num_workers: The number of workers for the dataloaders.
        seed: The seed for the random number generators.

    Returns:
        4 DataLoaders: train_lbl_dl, train_unl_dl, val_unl_dl, test_dl
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

    StratifiedShuffleSplit()

    train_size = len(src_train_ds)
    train_unl_size = int(train_size * train_unl_size)
    train_lbl_size = int(train_size * train_lbl_size)
    val_size = int(train_size - train_unl_size - train_lbl_size)

    targets = np.array(src_train_ds.targets)
    ixs = np.arange(len(targets))
    train_unl_ixs, lbl_ixs = train_test_split(
        ixs,
        train_size=train_unl_size,
        stratify=targets,
        random_state=seed,
    )
    lbl_targets = targets[lbl_ixs]

    val_ixs, train_lbl_ixs = train_test_split(
        lbl_ixs,
        train_size=val_size,
        stratify=lbl_targets,
        random_state=seed,
    )

    train_lbl_ds = Subset(src_train_ds, train_lbl_ixs)
    train_unl_ds = Subset(src_train_ds, train_unl_ixs)
    val_ds = Subset(src_train_ds, val_ixs)

    dl_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # We use drop_last=True to ensure that the batch size is always the same
    # This is crucial as we need to average the predictions across the batch
    # size axis.

    train_lbl_dl = DataLoader(train_lbl_ds, shuffle=True, **dl_args)
    train_unl_dl = DataLoader(train_unl_ds, shuffle=True, **dl_args)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_args)
    test_dl = DataLoader(src_test_ds, shuffle=False, **dl_args)

    return (
        train_lbl_dl,
        train_unl_dl,
        val_dl,
        test_dl,
        src_train_ds.classes,
    )
