import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class MNISTDataModule:
    def __init__(self, data_dir='./mnist_data', batch_size=32, train_transforms=None, val_transforms=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Default transforms if none provided
        self.train_transforms = train_transforms if train_transforms else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.val_transforms = val_transforms if val_transforms else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self):
        # Download and load training data
        full_train_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=self.train_transforms
        )
        
        # Split training data into train and validation
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )
        
        # Download and load test data
        self.test_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=self.val_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        ) 