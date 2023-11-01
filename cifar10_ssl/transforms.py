import torch
from torchvision import transforms
from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomCrop,
)

# from torchvision.transforms.v2 import AutoAugmentPolicy, AutoAugment

tf_preproc = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

tf_aug = transforms.Compose(
    [
        lambda x: torch.nn.functional.pad(
            x,
            (
                4,
                4,
                4,
                4,
            ),
            mode="reflect",
        ),
        RandomCrop(32),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
    ]
)
