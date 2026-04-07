# WAF

This repository contains the implementation of semi-supervised learning using **FixMatch** for ultrasound image classification across three different datasets: **CCAUI**, **DDTI**, and **HC18**.

## Project Structure
```text
├── __pycache__/              # Cached Python files
├── dataloaders/              # Data loading utilities for each dataset
├── networks/                 # Model architectures
├── utils/                    # Helper functions and utilities
├── test_CCAUI.py             # Testing script for CCAUI dataset
├── test_DDTI.py              # Testing script for DDTI dataset
├── test_HC18.py              # Testing script for HC18 dataset
├── train_fixmatch_CCAUI.py   # Training script for CCAUI using FixMatch
├── train_fixmatch_DDTI.py    # Training script for DDTI using FixMatch
└── train_fixmatch_HC18.py    # Training script for HC18 using FixMatch
```

## Datasets

- **CCAUI** - Carotid artery ultrasound image dataset
- **DDTI** - Thyroid nodule ultrasound image dataset  
- **HC18** - Fetal head circumference dataset

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Other dependencies (see `utils` for specific imports)

## Training

To train a model on a specific dataset:

```bash
# For CCAUI dataset
python train_fixmatch_CCAUI.py

# For DDTI dataset
python train_fixmatch_DDTI.py

# For HC18 dataset
python train_fixmatch_HC18.py
