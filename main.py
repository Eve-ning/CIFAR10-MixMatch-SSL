from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from cifar10_ssl.data import get_dataloaders
from cifar10_ssl.eval import evaluate, compute_loss
from cifar10_ssl.mix_match import mix_match
from cifar10_ssl.models import wide_resnet_50_2, update_ema

DEVICE = "cuda"

N_AUGS = 2
SHARPEN_TEMP = 0.5
N_EPOCHS = 2
N_TRAIN_ITERS = 1024
LR = 0.002
WEIGHT_DECAY = LR * 0.02

train_lbl_dl, train_unl_dl, val_dl, test_dl, classes = get_dataloaders(
    dataset_dir=(Path(__file__).parent / "data").as_posix(),
    train_lbl_size=0.005,
    train_unl_size=0.980,
    batch_size=64,
    num_workers=0,
    seed=42,
)
print(
    f"train_lbl_dl: {len(train_lbl_dl.dataset)}, "
    f"train_unl_dl: {len(train_unl_dl.dataset)}, "
    f"val_unl_dl: {len(val_dl.dataset)}, "
    f"test_dl: {len(test_dl.dataset)}, "
)
n_classes = len(classes)

net = wide_resnet_50_2(n_classes=n_classes).to(DEVICE)
ema_net = deepcopy(net).to(DEVICE)

optimizer = torch.optim.Adam(
    net.fc.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
lbl_loss_fn = nn.CrossEntropyLoss()
loss_unl_fn = nn.MSELoss()

loss_unl_scaler = np.linspace(0, 100, N_EPOCHS)
ema_decayer = np.linspace(0.9, 0.999, N_EPOCHS)

for epoch in (t_eval := tqdm(range(N_EPOCHS))):
    net.train()

    for train_ix in (t := tqdm(range(N_TRAIN_ITERS), leave=False)):
        for (x_unl, _), (x_lbl, y_lbl) in zip(train_unl_dl, train_lbl_dl):
            BS, CH, H, W = x_unl.shape
            CLS = n_classes

            optimizer.zero_grad()
            x_unl = x_unl.to(DEVICE)
            x_lbl = x_lbl.to(DEVICE)
            y_lbl = y_lbl.to(DEVICE)

            with torch.no_grad():
                (x_lbl_mix, y_lbl_mix), (x_unl_mix, y_unl_mix) = mix_match(
                    x_unl=x_unl,
                    x_lbl=x_lbl,
                    y_lbl=y_lbl,
                    n_augs=N_AUGS,
                    net=net,
                    sharpen_temp=SHARPEN_TEMP,
                )

            assert x_lbl_mix.shape == (BS, CH, H, W), x_lbl_mix.shape
            assert x_unl_mix.shape == (BS, N_AUGS, CH, H, W), x_unl_mix.shape
            assert y_lbl_mix.shape == (BS, CLS), y_lbl_mix.shape
            assert y_unl_mix.shape == (BS, N_AUGS, CLS), y_unl_mix.shape

            y_lbl_pred = net(x_lbl_mix)
            assert y_lbl_pred.shape == (BS, CLS), y_lbl_pred.shape

            y_unl_pred = torch.stack(
                [net(x_unl_mix[:, aug_i]) for aug_i in range(N_AUGS)],
                dim=1,
            )
            assert y_unl_pred.shape == (BS, N_AUGS, CLS), y_unl_pred.shape

            loss_lbl, loss_unl = compute_loss(
                y_lbl_pred=y_lbl_pred,
                y_lbl_tgt=y_lbl_mix,
                y_unl_pred=y_unl_pred,
                y_unl_tgt=y_unl_mix,
                lbl_loss_fn=lbl_loss_fn,
                unl_loss_fn=loss_unl_fn,
                unl_loss_scale=float(loss_unl_scaler[epoch]),
            )
            loss = loss_lbl + loss_unl

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                update_ema(net, ema_net, float(ema_decayer[epoch]))
                lbl_acc = (
                    (torch.argmax(y_lbl_pred, dim=1) == y_lbl).float().mean()
                )

            t.set_description(
                f"lbl.Loss: {loss_lbl:.2f}, unl.Loss: {loss_unl:.2f}, "
                f"lbl.Acc: {lbl_acc:.2%}"
            )

    val_acc = evaluate(net=ema_net, dl=val_dl, device=DEVICE)

    t_eval.set_description(f"Epoch: {epoch}, Val Acc: {val_acc:.2%}")
