import torch
import torch.optim as optim
from data.dataset import MNISTDataModule
from model.mnist_model import MNISTNet
from augmentation.transforms import get_train_transforms, get_val_transforms
from utils import train_epoch, evaluate
import os

def main():
    # Training settings
    batch_size = 64
    epochs = 10
    lr = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data module with augmentations
    data_module = MNISTDataModule(
        batch_size=batch_size,
        train_transforms=get_train_transforms(),
        val_transforms=get_val_transforms()
    )
    data_module.setup()
    
    # Initialize model
    model = MNISTNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Create directory for saving models
    # os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_accuracy = 0.0
    train_losses = []
    
    for epoch in range(1, epochs + 1):
        # Train and capture loss
        avg_train_loss = train_epoch(model, device, data_module.train_dataloader(), optimizer, epoch)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(model, device, data_module.val_dataloader(), "Validation")
        
        # Update best accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")
        
        # Save best model
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints/mnist_model_epoch_{epoch}.pt')
    
    # Final evaluation on test set
    test_loss, test_accuracy = evaluate(model, device, data_module.test_dataloader(), "Test")
    print(f"Final Test Results: Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.2f}%")

if __name__ == '__main__':
    main() 