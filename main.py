import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import (
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
)
from tqdm import tqdm

from cifar10_ssl.data import get_dataloaders
from cifar10_ssl.mix_match import mix_match

DEVICE = "cuda"

NUM_UNL_AUGS = 2
SHARPEN_TEMP = 0.5

train_lbl_dl, train_unl_dl, test_dl = get_dataloaders(
    dataset_dir=(Path(__file__).parent / "data").as_posix(),
    train_lbl_size=0.005,
    train_unl_size=0.005,
    test_size=0.01,
    batch_size=64,
    num_workers=0,
    seed=42,
)

# Alternatively, we can use ResNet50
# net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

# Freeze all layers except the last one
# for p in net.parameters():
#     p.requires_grad = False

# Alternatively, we can freeze only the first few layers
# for p in (net.layer1, net.layer2, net.layer3):
#     p.requires_grad = False

net.fc = nn.Sequential(
    nn.Linear(net.fc.in_features, 10),
    nn.Softmax(dim=1),
)
net.to(DEVICE)

ema_net = copy.deepcopy(net)


def update_ema_variables(model, ema_model, ema_decay):
    with torch.no_grad():
        for ema_param, param in zip(
            ema_model.parameters(), model.parameters()
        ):
            ema_param.data.copy_(
                ema_param.data * ema_decay + (1 - ema_decay) * param.data
            )


optimizer = torch.optim.Adam(
    net.fc.parameters(),
    lr=0.002,
    weight_decay=1e-4,
    amsgrad=True,
)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(
#     optimizer,
#     gamma=0.999,
# )
loss_lbl_fn = nn.CrossEntropyLoss()

# Alternatively, we can use KLDivLoss
# loss_unl_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
loss_unl_fn = nn.MSELoss()
loss_unl_scale = 100

n_epochs = 100

for epoch in (t := tqdm(range(n_epochs))):
    net.train()
    for (x_unl, _), (x_lbl, y_lbl) in zip(train_unl_dl, train_lbl_dl):
        BS, CH, H, W = x_unl.shape
        CLS = 10

        optimizer.zero_grad()
        x_unl = x_unl.to(DEVICE)
        x_lbl = x_lbl.to(DEVICE)
        y_lbl = y_lbl.to(DEVICE)

        with torch.no_grad():
            (x_lbl_mix, y_lbl_mix), (x_unl_mix, y_unl_mix) = mix_match(
                x_unl,
                x_lbl,
                y_lbl,
                n_augs=NUM_UNL_AUGS,
                net=net,
                sharpen_temp=SHARPEN_TEMP,
            )
        assert x_lbl_mix.shape == (BS, CH, H, W), x_lbl_mix.shape
        assert x_unl_mix.shape == (BS, NUM_UNL_AUGS, CH, H, W), x_unl_mix.shape
        assert y_lbl_mix.shape == (BS, CLS), y_lbl_mix.shape
        assert y_unl_mix.shape == (BS, NUM_UNL_AUGS, CLS), y_unl_mix.shape

        y_lbl_pred_logits = net(x_lbl_mix)
        assert y_lbl_pred_logits.shape == (BS, CLS), y_lbl_pred_logits.shape
        y_unl_pred_logits = torch.stack(
            [net(x_unl_mix[:, aug_i]) for aug_i in range(NUM_UNL_AUGS)], dim=1
        )
        assert y_unl_pred_logits.shape == (BS, NUM_UNL_AUGS, CLS), (
            y_unl_pred_logits.shape,
        )

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
        update_ema_variables(net, ema_net, 0.999)
        # scheduler.step()

        with torch.no_grad():
            lbl_acc = (
                (torch.argmax(y_lbl_pred_logits, dim=1) == y_lbl)
                .float()
                .mean()
            )

        t.set_description(
            # f"lr: {scheduler.get_last_lr()[0]:.4f}, "
            f"lbl.Loss: {loss_lbl:.2f}, unl.Loss: {loss_unl:.2f}, "
            f"lbl.Acc: {lbl_acc:.2%}"
        )

    net.eval()
    accs = []
    for x, y in test_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_pred_logits = net(x)
        y_pred = torch.argmax(y_pred_logits, dim=1)
        acc = (y_pred == y).float().mean().cpu().numpy()
        accs.append(acc)
    print(f"Test Acc: {np.mean(accs):.2%}")
