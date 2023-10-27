from torchvision import transforms

tf_preproc = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

tf_aug = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        # transforms.RandomRotation(degrees=15),
        # transforms.RandomResizedCrop(size=(32, 32), scale=(0.7, 1.0)),
    ]
)
