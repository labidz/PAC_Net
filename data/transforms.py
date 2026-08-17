"""
Image transforms for PAC-Net.

DINOv2 expects:  224×224, normalised with ImageNet mean/std.
Training augmentations are strong (standard for medical imaging CV).
Val/Test use only centre crop + normalisation.
"""

from torchvision import transforms

# DINOv2 / ImageNet normalisation
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0),
                                     ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def get_eval_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),     # 256 for img_size=224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
