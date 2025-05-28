from torchvision import transforms


def get_train_transforms():
    """
    Returns the transforms for training data.
    Includes various augmentations to improve model robustness.
    """
    return transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def get_val_transforms():
    """
    Returns the transforms for validation/test data.
    Only includes basic normalization.
    """
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
