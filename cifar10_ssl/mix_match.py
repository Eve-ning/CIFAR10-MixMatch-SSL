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
    n_lbl = x_lbl.shape[0]
    n_unl = x_unl.shape[0]
    x = torch.cat([x_lbl, x_unl], dim=0)
    y = torch.cat(
        [y_lbl, y_unl],
        dim=0,
    )
    perm = torch.randperm(x.shape[0])
    x_shuf = x[perm]
    y_shuf = y[perm]
    return (
        mix_up(x_lbl, y_lbl, x_shuf[:n_lbl], y_shuf[:n_lbl]),
        mix_up(x_unl, y_unl, x_shuf[n_lbl:], y_shuf[n_lbl:]),
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
    y_lbl_ohe = nn.functional.one_hot(y_lbl, num_classes=10)
    # x_lbl: [Batch Size, C, H, W]
    x_lbl_aug = tf_aug(x_lbl)
    # x_unl: No. Augs * [Batch Size, C, H, W]
    x_unl_aug: list[torch.Tensor] = [
        tf_aug(x_unl) for _ in range(n_augs)
    ]

    # This computes the prediction of each augmentation
    # then averages them.
    # y_unl_aug_pred_logits: [Batch Size, No. Classes]
    with torch.no_grad():
        y_unl_aug_pred_logits = torch.stack(list(map(net, x_unl_aug))).mean(
            dim=0
        )

    # y_unl_aug_pred_logits_sharpen: [Batch Size, No. Classes]
    y_unl_aug_pred_logits_sharpen = sharpen(
        y_unl_aug_pred_logits, sharpen_temp
    )

    # x: [Batch Size * (1 + No. Augs), C, H, W]
    # y: [Batch Size * (1 + No. Augs), No. Classes]

    return mix_up_partitioned(
        x_lbl=x_lbl_aug,
        x_unl=torch.cat(x_unl_aug, dim=0),
        y_lbl=y_lbl_ohe,
        y_unl=y_unl_aug_pred_logits_sharpen.repeat(n_augs, 1),
    )
