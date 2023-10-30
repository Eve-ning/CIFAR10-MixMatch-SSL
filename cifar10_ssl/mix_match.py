import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import (
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
    resnet50,
    ResNet50_Weights,
)
from tqdm import tqdm

from cifar10_ssl.data import get_dataloaders
from cifar10_ssl.transforms import tf_preproc, tf_aug


def sharpen(
    logits: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    logits_inv_temp = logits ** (1 / temp)
    return logits_inv_temp / logits_inv_temp.sum(dim=1, keepdim=True)


def mix_up(
    x: torch.Tensor,
    y: torch.Tensor,
    x_shuf: torch.Tensor,
    y_shuf: torch.Tensor,
    alpha: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)
    x = l * x + (1 - l) * x_shuf
    y = l * y + (1 - l) * y_shuf
    return x, y


def mix_up_partitioned(
    x_lbl: torch.Tensor,
    x_unl: torch.Tensor,
    y_lbl: torch.Tensor,
    y_unl: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]:
    BS, AUGS, CH, H, W = x_unl.shape
    CLS = y_lbl.shape[1]

    assert x_lbl.shape == (BS, CH, H, W), x_lbl.shape
    assert x_unl.shape == (BS, AUGS, CH, H, W), x_unl.shape
    assert y_lbl.shape == (BS, CLS), y_lbl.shape
    assert y_unl.shape == (BS, AUGS, CLS), y_unl.shape

    n_lbl = x_lbl.shape[0]
    x = torch.cat([torch.unsqueeze(x_lbl, dim=1), x_unl], dim=1)
    y = torch.cat([torch.unsqueeze(y_lbl, dim=1), y_unl], dim=1)
    assert x.shape == (BS, 1 + AUGS, CH, H, W), x.shape
    assert y.shape == (BS, 1 + AUGS, CLS), y.shape

    # We only shuffle on the Augmentations axis
    perm = torch.randperm(x.shape[1])
    x_shuf = x[:, perm]
    y_shuf = y[:, perm]
    assert x_shuf.shape == (BS, 1 + AUGS, CH, H, W), x_shuf.shape
    assert y_shuf.shape == (BS, 1 + AUGS, CLS), y_shuf.shape

    return (
        mix_up(x_lbl, y_lbl, x_shuf[:, 0], y_shuf[:, 0]),
        mix_up(x_unl, y_unl, x_shuf[:, 1:], y_shuf[:, 1:]),
    )


def mix_match(
    x_unl: torch.Tensor,
    x_lbl: torch.Tensor,
    y_lbl: torch.Tensor,
    n_augs: int,
    net: nn.Module,
    sharpen_temp: float = 0.5,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]:
    BS, CH, H, W = x_unl.shape
    CLS = 10

    assert x_lbl.shape == (BS, CH, H, W), x_lbl.shape
    assert y_lbl.shape == (BS,), y_lbl.shape

    y_lbl_ohe = nn.functional.one_hot(y_lbl, num_classes=CLS)
    assert y_lbl_ohe.shape == (BS, CLS), y_lbl_ohe.shape

    x_lbl_aug = tf_aug(x_lbl)
    assert x_lbl_aug.shape == (BS, CH, H, W), x_lbl_aug.shape

    x_unl_aug: torch.Tensor = torch.stack(
        [tf_aug(x_unl) for _ in range(n_augs)], dim=1
    )
    assert x_unl_aug.shape == (BS, n_augs, CH, H, W), x_unl_aug.shape

    # This computes the prediction of each augmentation
    # then averages them.
    with torch.no_grad():
        y_unl_aug_pred_logits = torch.stack(
            [net(x_unl_aug[:, aug_i]) for aug_i in range(n_augs)], dim=1
        )
        assert y_unl_aug_pred_logits.shape == (BS, n_augs, CLS), (
            y_unl_aug_pred_logits.shape,
        )

        y_unl_aug_pred_logits_mean = y_unl_aug_pred_logits.mean(dim=1)
        assert y_unl_aug_pred_logits_mean.shape == (BS, CLS), (
            y_unl_aug_pred_logits_mean.shape,
        )

        y_unl_aug_pred_logits_sharpen = sharpen(
            y_unl_aug_pred_logits_mean, sharpen_temp
        )
        assert y_unl_aug_pred_logits_sharpen.shape == (BS, CLS), (
            y_unl_aug_pred_logits_sharpen.shape,
        )

    # We need to repeat to match the shape of the x_unl_aug
    y_unl_repeat = torch.unsqueeze(
        y_unl_aug_pred_logits_sharpen, dim=1
    ).repeat(1, n_augs, 1)
    assert y_unl_repeat.shape == (BS, n_augs, CLS), y_unl_repeat.shape

    return mix_up_partitioned(
        x_lbl=x_lbl_aug,
        x_unl=x_unl_aug,
        y_lbl=y_lbl_ohe,
        y_unl=y_unl_repeat,
    )
