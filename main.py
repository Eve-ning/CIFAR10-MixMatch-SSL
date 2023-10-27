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

NUM_UNL_AUGS = 2
SHARPEN_TEMP = 0.5

train_lbl_dl, train_unl_dl, test_dl = get_dataloaders(
    dataset_dir=(Path(__file__).parent / "data").as_posix(),
    train_lbl_size=0.05,
    train_unl_size=0.05,
    test_size=0.01,
    batch_size=256,
    num_workers=0,
    seed=42
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
    nn.Linear(net.fc.in_features, 10),
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
            x_unl, x_lbl, y_lbl,
            n_augs=NUM_UNL_AUGS,
            net=net,
            sharpen_temp=SHARPEN_TEMP,
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
