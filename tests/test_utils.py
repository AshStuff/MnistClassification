import io  # For capturing print output
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import evaluate  # noqa: E402


class SimpleTestModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return torch.log_softmax(self.fc(x), dim=1)


class TestUtils(unittest.TestCase):
    def test_evaluate_function_returns_correct_types(self):
        """Test that evaluate function returns loss and accuracy with correct types."""
        # Create a simple model and data
        model = SimpleTestModel()
        device = torch.device("cpu")
        
        # Create mock data loader with simple data
        mock_data_loader = MagicMock()
        
        # Create simple test data
        data = torch.randn(4, 10)  # 4 samples, 10 features
        targets = torch.tensor([0, 1, 0, 1])  # Simple binary targets
        
        mock_data_loader.__iter__.return_value = iter([(data, targets)])
        mock_data_loader.dataset = MagicMock()
        mock_data_loader.dataset.__len__.return_value = 4
        
        # Capture stdout to avoid cluttering test output
        captured_output = io.StringIO()
        with patch("sys.stdout", new=captured_output):
            loss, accuracy = evaluate(model, device, mock_data_loader, "Test")
        
        # Check that function returns correct types
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        
        # Check that values are reasonable
        self.assertGreater(loss, 0)  # Loss should be positive
        self.assertGreaterEqual(accuracy, 0)  # Accuracy should be >= 0
        self.assertLessEqual(accuracy, 100)  # Accuracy should be <= 100
        
        # Check that output was printed
        output_str = captured_output.getvalue()
        self.assertIn("Test set:", output_str)
        self.assertIn("Average loss:", output_str)
        self.assertIn("Accuracy:", output_str)


if __name__ == "__main__":
    unittest.main()
