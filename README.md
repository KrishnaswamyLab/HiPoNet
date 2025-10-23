
# HiPoNet

This repository is for HiPoNet, a method designed to learn from high-dimensional point cloud data using multiple graph embeddings and graph wavelet transforms.
![HiPoNet Pipeline](archive/graphical_abstract.png)
## Overview

The provided Python script trains HiPoNet on a given dataset of point features and associated labels. It uses:

- **PyTorch** for model definition and training.
- **wandb** (Weights & Biases) for experiment tracking.
- **scikit-learn** for data splitting.
- **tqdm** for progress visualization.

The script:
1. Loads and processes your input data.
2. Initializes the model and MLP classifier.
3. Trains the model, logging training progress and metrics to `wandb`.
4. Saves the best model weights for reproducibility.

## Requirements

- Python 3.7 or later
- PyTorch (compatible with CUDA if GPU training is desired)
- NumPy
- scikit-learn
- tqdm
- wandb

Install all requirements using:
```bash
pip install torch numpy scikit-learn tqdm wandb
```

## Arguments

You can specify various arguments to customize training:

- `--raw_dir` (str): Directory containing the raw data. Default: `melanoma_data_full`
- `--full` (flag): If provided, may indicate the use of a full dataset variant.
- `--num_weights` (int): Number of weights (features dimensions) to learn. Default: 2
- `--threshold` (float): Threshold used for graph creation. Default: `5e-5`
- `--hidden_dim` (int): Hidden dimension size for the MLP. Default: `50`
- `--num_layers` (int): Number of MLP layers. Default: `3`
- `--lr` (float): Learning rate. Default: `0.03`
- `--wd` (float): Weight decay. Default: `3e-3`
- `--num_epochs` (int): Number of training epochs. Default: `100`
- `--batch_size` (int): Batch size for training. Default: `128`
- `--gpu` (int): GPU index to use. Set to `-1` for CPU-only. Default: `0`

## Running the Script

Before running, ensure that `raw_dir` points to a directory containing compatible data files. The data loading and preparation code is assumed to be handled within the `PointCloudFeatLearning` class. Consult that class for specifics on required data format.

Run the script:
```bash
python train_pointcloudnet.py --raw_dir path_to_data --num_weights 2 --threshold 0.00005 --gpu 0
```

Adjust parameters as needed. For example:
- To train on CPU:
    ```bash
    python train_pointcloudnet.py --gpu -1
    ```
- To change the learning rate and number of epochs:
    ```bash
    python train_pointcloudnet.py --lr 0.01 --num_epochs 200
    ```

## Weights & Biases Integration

The script automatically logs metrics to [Weights & Biases](https://wandb.ai/) if you have an account and have run `wandb login` locally. If you do not want to use `wandb`, remove or comment out the `wandb` lines in the code.

## Outputs

- **Model Checkpoints:** The best performing model weights will be saved as `bestalpha_{num_weights}` and `bestmlp_{num_weights}`.
- **Alpha Weights:** The learned alpha weights for feature importance are saved as `bestweights_{num_weights}.pt`.

These files can be used to reproduce results or for downstream analysis.
```
