import copy

from torch import nn
from torchvision.models import (
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
)


def wide_resnet_50_2(n_classes=10) -> nn.Module:
    net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, n_classes),
    )
    return net


def update_ema(
    net: nn.Module,
    ema_net: nn.Module,
    ema_alpha: float,
):
    """Update the EMA network.

    Notes:
        The alpha value is used to control the weight of the EMA network.

    Args:
        net: The network to use for the parameters.
        ema_net: The EMA network to update.
        ema_alpha: The alpha value for the EMA.
    """

    for ema_param, net_param in zip(ema_net.parameters(), net.parameters()):
        ema_param.data.mul_(ema_alpha).add_(
            net_param.data, alpha=(1 - ema_alpha)
        )
        # net_param.mul_(1 - (0.002 * 0.02))
