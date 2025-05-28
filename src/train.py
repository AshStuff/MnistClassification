import torch
import torch.optim as optim

from augmentation.transforms import get_train_transforms, get_val_transforms
from data.dataset import MNISTDataModule
from model.mnist_model import MNISTNet
from utils import evaluate, train_epoch


def main():
    # Training settings
    batch_size = 64
    epochs = 10
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data module with augmentations
    data_module = MNISTDataModule(
        batch_size=batch_size,
        train_transforms=get_train_transforms(),
        val_transforms=get_val_transforms(),
    )
    data_module.setup()

    # Initialize model
    model = MNISTNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Create directory for saving models
    # os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_epoch(model, device, data_module.train_dataloader(), optimizer, epoch)

        # Evaluate on validation set
        evaluate(model, device, data_module.val_dataloader(), "Validation")

        # Save best model
        # if epoch % 5 == 0:
        #     torch.save(model.state_dict(), f"checkpoints/mnist_model_epoch_{epoch}.pt")

    # Final evaluation on test set
    evaluate(model, device, data_module.test_dataloader(), "Test")


if __name__ == "__main__":
    main()
