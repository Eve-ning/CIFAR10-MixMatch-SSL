from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    net: nn.Module,
    dl: DataLoader,
    device: torch.device | str,
) -> float:
    accs = []
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        y_pred_logits = net(x)
        y_pred = torch.argmax(y_pred_logits, dim=1)
        acc = (y_pred == y).float().mean().cpu().numpy()
        accs.append(acc)
    return np.mean(accs)


def compute_loss(
    y_lbl_pred,
    y_lbl_tgt,
    y_unl_pred,
    y_unl_tgt,
    lbl_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    unl_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    unl_loss_scale: float,
):
    loss_lbl = lbl_loss_fn(y_lbl_pred, y_lbl_tgt)
    loss_unl = (
        unl_loss_fn(
            nn.functional.softmax(y_unl_pred, dim=1),
            nn.functional.softmax(y_unl_tgt, dim=1),
        )
        * unl_loss_scale
    )

    return loss_lbl, loss_unl
