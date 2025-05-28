import os
import sys
import unittest

import numpy as np
import torch
from PIL import Image

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from augmentation.transforms import (  # noqa: E402
    get_train_transforms,
    get_val_transforms,
)


class TestAugmentationTransforms(unittest.TestCase):
    def _create_dummy_pil_image(self, size=(32, 32), mode="L"):
        """Helper to create a dummy PIL Image."""
        # 'L' mode for grayscale, matching common MNIST-like datasets
        # The normalization constants (0.1307,), (0.3081,) suggest single-channel.
        return Image.fromarray(np.uint8(np.random.rand(*size) * 255), mode=mode)

    def test_get_train_transforms_output_is_float(self):
        """Test that train transforms output a float tensor."""
        dummy_image = self._create_dummy_pil_image()
        train_transforms = get_train_transforms()

        transformed_tensor = train_transforms(dummy_image)

        self.assertIsInstance(
            transformed_tensor, torch.Tensor, "Output should be a torch.Tensor"
        )
        self.assertEqual(
            transformed_tensor.dtype, torch.float32, "Tensor dtype should be float32"
        )

        # Basic shape check: C, H, W. For grayscale, C=1.
        self.assertEqual(
            transformed_tensor.ndim, 3, "Tensor should have 3 dimensions (C, H, W)"
        )
        self.assertEqual(
            transformed_tensor.shape[0],
            1,
            "Channel dimension should be 1 for grayscale",
        )
        self.assertEqual(
            transformed_tensor.shape[1], 32, "Height should match dummy image"
        )
        self.assertEqual(
            transformed_tensor.shape[2], 32, "Width should match dummy image"
        )

    def test_get_val_transforms_output_is_float(self):
        """Test that validation transforms output a float tensor."""
        dummy_image = self._create_dummy_pil_image()
        val_transforms = get_val_transforms()

        transformed_tensor = val_transforms(dummy_image)

        self.assertIsInstance(
            transformed_tensor, torch.Tensor, "Output should be a torch.Tensor"
        )
        self.assertEqual(
            transformed_tensor.dtype, torch.float32, "Tensor dtype should be float32"
        )

        # Basic shape check: C, H, W. For grayscale, C=1.
        self.assertEqual(
            transformed_tensor.ndim, 3, "Tensor should have 3 dimensions (C, H, W)"
        )
        self.assertEqual(
            transformed_tensor.shape[0],
            1,
            "Channel dimension should be 1 for grayscale",
        )
        self.assertEqual(
            transformed_tensor.shape[1], 32, "Height should match dummy image"
        )
        self.assertEqual(
            transformed_tensor.shape[2], 32, "Width should match dummy image"
        )


if __name__ == "__main__":
    unittest.main()
