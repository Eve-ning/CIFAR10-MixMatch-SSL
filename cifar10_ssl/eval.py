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
    """Evaluate the network on the given dataloader.

    Args:
        net: The network to evaluate.
        dl: The dataloader to use.
        device: The device to use.

    Returns:
        The accuracy of the network on the dataloader.
    """
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the loss for the labelled and unlabelled data.

    Args:
        y_lbl_pred: The prediction for the labelled data.
        y_lbl_tgt: The target for the labelled data.
        y_unl_pred: The prediction for the unlabelled data.
        y_unl_tgt: The target for the unlabelled data.
        lbl_loss_fn: The loss function for the labelled data.
        unl_loss_fn: The loss function for the unlabelled data.
        unl_loss_scale: The scale for the unlabelled loss.

    Returns:
        A tuple of the labelled and unlabelled losses.
    """

    loss_lbl = lbl_loss_fn(y_lbl_pred, y_lbl_tgt)
    loss_unl = (
        unl_loss_fn(
            nn.functional.softmax(y_unl_pred, dim=1),
            nn.functional.softmax(y_unl_tgt, dim=1),
        )
        * unl_loss_scale
    )

    return loss_lbl, loss_unl
