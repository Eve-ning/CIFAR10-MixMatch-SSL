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

from cifar10_ssl.transforms import tf_preproc, tf_aug

DATASET_DIR = (Path(__file__).parent / "data").as_posix()
NUM_WORKERS = 0
NUM_UNL_AUGS = 2
TRAIN_LBL_SIZE = 0.05
TRAIN_ULB_SIZE = 0.05
BATCH_SIZE = 256
TEST_LBL_SIZE = 0.01
DISCARD_LBL_SIZE = 1 - TRAIN_LBL_SIZE - TRAIN_ULB_SIZE - TEST_LBL_SIZE
SEED = 42
SHARPEN_TEMP = 0.5

src_train_ds = CIFAR10(
    DATASET_DIR,
    train=True,
    download=True,
    transform=tf_preproc,
)
src_test_ds = CIFAR10(
    DATASET_DIR,
    train=False,
    download=True,
    transform=tf_preproc,
)
classes = src_train_ds.classes
ohe = OneHotEncoder().fit([classes])
n_classes = len(classes)

train_lbl_ds, train_unl_ds, test_ds, _ = random_split(
    src_train_ds,
    [TRAIN_LBL_SIZE, TRAIN_ULB_SIZE, TEST_LBL_SIZE, DISCARD_LBL_SIZE],
    generator=torch.Generator().manual_seed(SEED),
)

# We use drop_last=True to ensure that the batch size is always the same
# This is crucial as we need to average the predictions across the batch
# size axis.
train_lbl_dl = DataLoader(
    train_lbl_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

train_unl_dl = DataLoader(
    train_unl_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)


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
) -> tuple[
    tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]:
    y_lbl_ohe = nn.functional.one_hot(y_lbl, num_classes=10)
    # x_lbl: [Batch Size, C, H, W]
    x_lbl_aug = tf_aug(x_lbl)
    # x_unl: No. Augs * [Batch Size, C, H, W]
    x_unl_aug: list[torch.Tensor] = [
        tf_aug(x_unl) for _ in range(NUM_UNL_AUGS)
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
        y_unl_aug_pred_logits, SHARPEN_TEMP
    )

    # x: [Batch Size * (1 + No. Augs), C, H, W]
    # y: [Batch Size * (1 + No. Augs), No. Classes]

    return mix_up_partitioned(
        x_lbl=x_lbl_aug,
        x_unl=torch.cat(x_unl_aug, dim=0),
        y_lbl=y_lbl_ohe,
        y_unl=y_unl_aug_pred_logits_sharpen.repeat(NUM_UNL_AUGS, 1),
    )


# Alternatively, we can use ResNet50
# net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

# Freeze all layers except the last one
for p in net.parameters():
    p.requires_grad = False

# Alternatively, we can freeze only the first few layers
# for p in (net.layer1, net.layer2, net.layer3):
#     p.requires_grad = False

net.fc = nn.Sequential(
    nn.Linear(net.fc.in_features, n_classes),
)
net.to("cuda")


optimizer = torch.optim.Adam(
    net.fc.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    amsgrad=True,
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    epochs=100,
    steps_per_epoch=len(train_lbl_dl),
    pct_start=0.2,
    anneal_strategy="linear",
    cycle_momentum=False,
)
loss_lbl_fn = nn.CrossEntropyLoss()

# Alternatively, we can use KLDivLoss
# loss_unl_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
loss_unl_fn = nn.MSELoss()
loss_unl_scale = 100

n_epochs = 100

for epoch in (t := tqdm(range(n_epochs))):
    net.train()
    for (x_unl, _), (x_lbl, y_lbl) in zip(train_unl_dl, train_lbl_dl):
        optimizer.zero_grad()
        x_unl = x_unl.to("cuda")
        x_lbl = x_lbl.to("cuda")
        y_lbl = y_lbl.to("cuda")
        (x_lbl_mix, y_lbl_mix), (x_unl_mix, y_unl_mix) = mix_match(
            x_unl, x_lbl, y_lbl
        )
        y_lbl_pred_logits = net(x_lbl_mix)
        y_unl_pred_logits = net(x_unl_mix)

        loss_lbl = loss_lbl_fn(y_lbl_pred_logits, y_lbl_mix)
        loss_unl = (
            loss_unl_fn(
                nn.functional.softmax(y_unl_pred_logits, dim=1),
                nn.functional.softmax(y_unl_mix, dim=1),
            )
            * loss_unl_scale
        )

        loss = loss_lbl + loss_unl

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            lbl_acc = (
                (torch.argmax(y_lbl_pred_logits, dim=1) == y_lbl)
                .float()
                .mean()
            )

        t.set_description(
            f"lr: {scheduler.get_last_lr()[0]:.4f}, "
            f"lbl.Loss: {loss_lbl:.2f}, unl.Loss: {loss_unl:.2f}, "
            f"lbl.Acc: {lbl_acc:.2%}"
        )

    net.eval()
    accs = []
    for x, y in test_dl:
        x = x.to("cuda")
        y = y.to("cuda")
        y_pred_logits = net(x)
        y_pred = torch.argmax(y_pred_logits, dim=1)
        acc = (y_pred == y).float().mean().cpu().numpy()
        accs.append(acc)
    print(f"Test Acc: {np.mean(accs):.2%}")
