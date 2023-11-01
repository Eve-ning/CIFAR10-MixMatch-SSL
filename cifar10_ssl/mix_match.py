import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax

from cifar10_ssl.transforms import tf_aug


def sharpen(
    logits: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    """Sharpen the logits

    Note:
        This function doesn't require Softmax to be applied to the logits
        as it is always scaled to sum to 1.

    Args:
        logits: The logits to sharpen.
        temp: The temperature to sharpen with.
    """
    logits_inv_temp = logits ** (1 / temp)
    return logits_inv_temp / logits_inv_temp.sum(dim=1, keepdim=True)


def mix_up(
    x: torch.Tensor,
    y: torch.Tensor,
    x_shuf: torch.Tensor,
    y_shuf: torch.Tensor,
    alpha: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup the images and labels

    Notes:
        This performs the MixUp operation on the images and labels.
        A ratio is drawn from a beta distribution, and the images
        and labels are mixed with that ratio.

        The ratio drawn is always greater than 0.5, and the larger ratio is
        always used for the labelled images, thus the labelled images are
        always more prevalent in the mixup.

    Args:
        x: The images.
        y: The labels.
        x_shuf: The shuffled images.
        y_shuf: The shuffled labels.
        alpha: The alpha parameter for the beta distribution.
    """
    ratio = np.random.beta(alpha, alpha)
    ratio = max(ratio, 1 - ratio)
    x = ratio * x + (1 - ratio) * x_shuf
    y = ratio * y + (1 - ratio) * y_shuf
    return x, y


def mix_up_partitioned(
    x_lbl: torch.Tensor,
    x_unl: torch.Tensor,
    y_lbl: torch.Tensor,
    y_unl: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]:
    """Mixup the images and labels, but partitioned

    Notes:
        This partitioning is necessary to avoid mixing up the order of the
        labelled and unlabelled images.

        We can visualize this, take for example the following:

        | Original  | Shuffled  | Mixup                     |
        |-----------|-----------|---------------------------|
        | LBL       | UNL1      | LBL * r + UNL1 * (1 - r)  |
        | UNL1      | LBL       | UNL1 * r + LBL * (1 - r)  |
        | UNL2      | UNL4      | UNL2 * r + UNL4 * (1 - r) |
        | UNL3      | UNL2      | UNL3 * r + UNL2 * (1 - r) |
        | UNL4      | UNL3      | UNL4 * r + UNL3 * (1 - r) |

        We create a shuffled version of the images, then mix by applying
        a ratio to the original and shuffled images with the parameter r, which
        is drawn from a beta distribution.

        Both the image, and the labels are mixed:
        - In the case of the image, the image values are simply mixed.
        - In the case of the labels, the labels are mixed by applying the
          ratio to the one-hot encoded labels.


    Args:
        x_lbl: The labelled images.
        x_unl: The unlabelled images.
        y_lbl: The labelled labels.
        y_unl: The unlabelled labels.

    """

    BS, AUGS, CH, H, W = x_unl.shape
    CLS = y_lbl.shape[1]

    assert x_lbl.shape == (BS, CH, H, W), x_lbl.shape
    assert x_unl.shape == (BS, AUGS, CH, H, W), x_unl.shape
    assert y_lbl.shape == (BS, CLS), y_lbl.shape
    assert y_unl.shape == (BS, AUGS, CLS), y_unl.shape

    x = torch.cat([torch.unsqueeze(x_lbl, dim=1), x_unl], dim=1)
    y = torch.cat([torch.unsqueeze(y_lbl, dim=1), y_unl], dim=1)
    assert x.shape == (BS, 1 + AUGS, CH, H, W), x.shape
    assert y.shape == (BS, 1 + AUGS, CLS), y.shape

    # We only shuffle on the Batch axis
    perm = torch.randperm(x.shape[0])
    x_shuf = x[perm]
    y_shuf = y[perm]
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
    """MixMatch

    Notes:
        This is the MixMatch algorithm, as described in the paper.

        We apply the following steps:
        1. Augment the labelled and unlabelled images.
        2. Compute the prediction of each augmentation of the unlabelled images.
        3. Average the predictions of the unlabelled images.
        4. Sharpen the averaged predictions.
        5. Repeat the sharpened predictions to match the number of augmentations.
        6. Mixup the labelled and unlabelled images and labels.
        7. Return the mixed up images and labels.

    Args:
        x_unl: The unlabelled images.
        x_lbl: The labelled images.
        y_lbl: The labelled labels.
        n_augs: The number of augmentations to apply.
        net: The network to use for the predictions.
        sharpen_temp: The temperature to sharpen the predictions with.
    """

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
    y_unl_aug_pred_logits = torch.stack(
        [softmax(net(x_unl_aug[:, aug_i]), dim=-1) for aug_i in range(n_augs)],
        dim=1,
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
