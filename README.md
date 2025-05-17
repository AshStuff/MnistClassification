# MNIST Classification with PyTorch

This project implements a CNN-based MNIST digit classification using PyTorch. The project is structured with separate modules for data handling, model architecture, and data augmentation.

## Project Structure
```
.
├── src/                # Source code directory
│   ├── data/          # Data handling utilities
│   ├── model/         # Model architecture definition
│   ├── augmentation/  # Data augmentation utilities
│   ├── train.py       # Training script
│   └── utils.py       # Utility functions
├── pyproject.toml     # Project dependencies
└── README.md         # This file
```

## Setup
1. Install the required packages:
```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Training
To train the model, run:
```bash
cd src
python train.py
``` 


The model will be trained on the MNIST dataset with the specified augmentations. Checkpoints will be saved in the `checkpoints` directory.
