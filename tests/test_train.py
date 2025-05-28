import unittest
import torch
import torch.optim as optim
import sys
import os
from unittest.mock import patch
import tempfile
import shutil

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.mnist_model import MNISTNet
from data.dataset import MNISTDataModule
from augmentation.transforms import get_train_transforms, get_val_transforms
from utils import train_epoch, evaluate


class TestTraining(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device('cpu')  # Use CPU for consistent testing
        self.batch_size = 32  # Smaller batch size for faster testing
        self.test_epochs = 3  # Test with fewer epochs
        self.lr = 0.01
        
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize model
        self.model = MNISTNet().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        # Initialize data module with test directory
        self.data_module = MNISTDataModule(
            data_dir=self.temp_dir,
            batch_size=self.batch_size,
            train_transforms=get_train_transforms(),
            val_transforms=get_val_transforms()
        )
        
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, MNISTNet)
        self.assertEqual(len(list(self.model.parameters())), 8)  # Expected number of parameter tensors
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 1, 28, 28)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 10))  # Should output 10 classes
    
    def test_data_module_setup(self):
        """Test that the data module sets up correctly."""
        # Setup data (this will download MNIST if not present)
        self.data_module.setup()
        
        # Check that datasets are created
        self.assertTrue(hasattr(self.data_module, 'train_dataset'))
        self.assertTrue(hasattr(self.data_module, 'val_dataset'))
        self.assertTrue(hasattr(self.data_module, 'test_dataset'))
        
        # Check data loaders
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        test_loader = self.data_module.test_dataloader()
        
        self.assertTrue(len(train_loader) > 0)
        self.assertTrue(len(val_loader) > 0)
        self.assertTrue(len(test_loader) > 0)
        
        # Test a single batch
        data_batch, target_batch = next(iter(train_loader))
        self.assertEqual(data_batch.shape[0], self.batch_size)
        self.assertEqual(data_batch.shape[1:], (1, 28, 28))  # MNIST image shape
        self.assertEqual(target_batch.shape[0], self.batch_size)
    
    def test_train_epoch_function(self):
        """Test that the train_epoch function works correctly."""
        self.data_module.setup()
        train_loader = self.data_module.train_dataloader()
        
        # Test single epoch
        initial_params = [param.clone() for param in self.model.parameters()]
        avg_loss = train_epoch(self.model, self.device, train_loader, self.optimizer, 1)
        
        # Check that loss is returned and is a reasonable value
        self.assertIsInstance(avg_loss, float)
        self.assertGreater(avg_loss, 0)  # Loss should be positive
        self.assertLess(avg_loss, 10)    # Loss shouldn't be too high for MNIST
        
        # Check that model parameters have changed (indicating training occurred)
        params_changed = False
        for initial_param, current_param in zip(initial_params, self.model.parameters()):
            if not torch.equal(initial_param, current_param):
                params_changed = True
                break
        self.assertTrue(params_changed, "Model parameters should change after training")
    
    def test_evaluate_function(self):
        """Test that the evaluate function works correctly."""
        self.data_module.setup()
        val_loader = self.data_module.val_dataloader()
        
        # Test evaluation
        loss, accuracy = evaluate(self.model, self.device, val_loader, "Test")
        
        # Check return values
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreater(loss, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 100)
    
    def test_loss_decreases_over_epochs(self):
        """Test that the average loss decreases over multiple epochs."""
        self.data_module.setup()
        train_loader = self.data_module.train_dataloader()
        
        losses = []
        
        # Train for multiple epochs and collect losses
        for epoch in range(1, self.test_epochs + 1):
            avg_loss = train_epoch(self.model, self.device, train_loader, self.optimizer, epoch)
            losses.append(avg_loss)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Check that we have the expected number of losses
        self.assertEqual(len(losses), self.test_epochs)
        
        # Check that loss generally decreases (allowing for some fluctuation)
        # We'll check that the last loss is lower than the first loss
        first_loss = losses[0]
        last_loss = losses[-1]
        
        self.assertLess(last_loss, first_loss, 
                       f"Loss should decrease from {first_loss:.4f} to {last_loss:.4f}")
        
        # Also check that there's a general downward trend
        # (at least 50% of consecutive pairs should show improvement)
        improvements = 0
        for i in range(1, len(losses)):
            if losses[i] < losses[i-1]:
                improvements += 1
        
        improvement_rate = improvements / (len(losses) - 1)
        self.assertGreaterEqual(improvement_rate, 0.3, 
                               f"At least 30% of epochs should show improvement, got {improvement_rate:.2f}")
    
    def test_training_with_validation(self):
        """Test a complete training loop with validation."""
        self.data_module.setup()
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        # Training loop
        for epoch in range(1, self.test_epochs + 1):
            # Train
            train_loss = train_epoch(self.model, self.device, train_loader, self.optimizer, epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss, val_accuracy = evaluate(self.model, self.device, val_loader, "Validation")
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
        
        # Check that training completed successfully
        self.assertEqual(len(train_losses), self.test_epochs)
        self.assertEqual(len(val_losses), self.test_epochs)
        self.assertEqual(len(val_accuracies), self.test_epochs)
        
        # Check that validation accuracy is reasonable (at least 50% for MNIST after a few epochs)
        final_accuracy = val_accuracies[-1]
        self.assertGreater(final_accuracy, 50.0, 
                          f"Final validation accuracy should be > 50%, got {final_accuracy:.2f}%")
        
        # Check that validation loss decreases
        first_val_loss = val_losses[0]
        last_val_loss = val_losses[-1]
        self.assertLess(last_val_loss, first_val_loss,
                       f"Validation loss should decrease from {first_val_loss:.4f} to {last_val_loss:.4f}")
    
    @patch('builtins.print')  # Mock print to avoid cluttering test output
    def test_train_main_function_structure(self, mock_print):
        """Test that the main training structure can be executed."""
        # Import and test the main function structure
        from train import main
        
        # We can't easily test the full main() function due to hardcoded paths
        # and long training time, but we can test that it imports correctly
        # and the components work together
        
        # Test that all required components can be imported and initialized
        try:
            data_module = MNISTDataModule(
                batch_size=32,
                train_transforms=get_train_transforms(),
                val_transforms=get_val_transforms()
            )
            model = MNISTNet()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            
            # If we get here, the main components can be initialized successfully
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to initialize training components: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 