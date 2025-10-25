# EEG Transformer Classification

A PyTorch implementation of a Vision Transformer (ViT) adapted for EEG signal classification.

## Project Structure
```
├── eeg_transformer.py     # Main transformer model and training code
├── transformer.py         # Original transformer implementation
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Features
- Loads EEG data from NPZ files
- Implements Vision Transformer architecture for EEG classification
- Includes channel attention mechanism
- Supports train/test splitting and normalization
- Saves best model during training

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Prepare your EEG data in NPZ format with:
   - X: EEG epochs, shape (n_epochs, n_channels, n_timepoints)
   - y: labels, shape (n_epochs,)

2. Run training:
```bash
python eeg_transformer.py
```

## Model Architecture
- Input shape: (batch_size, 1, n_channels, n_timepoints)
- Components:
  - Channel Attention
  - Patch Embedding
  - Transformer Encoder
  - Classification Head
