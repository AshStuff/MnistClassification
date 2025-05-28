import unittest
import torch
from unittest.mock import MagicMock, patch
import io # For capturing print output

# Adjust the import path based on your project structure
# If src is a package and in PYTHONPATH, this should work.
# Otherwise, you might need to adjust sys.path in a test runner setup.
from src.utils import evaluate

class TestUtils(unittest.TestCase):
    def test_evaluate_accuracy_bug(self):
        # Mock model
        mock_model = MagicMock()
        # Simulate model output: 2 batches
        # Batch 1: 2 correct out of 3
        # Batch 2: 1 correct out of 2
        # Total: 3 correct out of 5
        mock_model.side_effect = [
            torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]), # Predictions for batch 1
            torch.tensor([[0.6, 0.4], [0.2, 0.8]])          # Predictions for batch 2
        ]

        # Mock device
        mock_device = torch.device('cpu')

        # Mock data_loader
        mock_data_loader = MagicMock()
        # Batch 1 data and targets
        data1 = torch.randn(3, 10) # 3 samples, 10 features
        target1 = torch.tensor([1, 0, 1]) # True labels for batch 1
        # Batch 2 data and targets
        data2 = torch.randn(2, 10) # 2 samples, 10 features
        target2 = torch.tensor([0, 0]) # True labels for batch 2 (model gets 2nd one wrong)

        mock_data_loader.__iter__.return_value = iter([
            (data1, target1),
            (data2, target2)
        ])
        mock_data_loader.dataset = MagicMock()
        mock_data_loader.dataset.__len__.return_value = 5 # Total number of samples

        # Capture stdout to check the print output
        captured_output = io.StringIO()
        with patch('sys.stdout', new=captured_output):
            # Call evaluate
            # Note: evaluate currently does not return accuracy, we check the print.
            # And it has a bug where the 'accuracy' variable for formatting is always 0.
            evaluate(mock_model, mock_device, mock_data_loader, set_name="Test")
        
        output_str = captured_output.getvalue()
        
        # Check if the print output shows the correct numbers for correct/total
        self.assertIn("Accuracy: 3/5", output_str)
        # Check if the print output shows the bugged percentage (always 0.00% due to accuracy variable not being updated)
        self.assertIn("(0.00%)", output_str)

if __name__ == '__main__':
    unittest.main() 