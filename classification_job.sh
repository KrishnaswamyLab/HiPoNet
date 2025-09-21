#!/bin/bash

#SBATCH --job-name=hiponet_classification
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.long@yale.edu
#SBATCH --gpus=h200:1
#SBATCH --partition=gpu_h200

source .venv/bin/activate
uv run wandb login $WANDB_API_KEY
echo "Starting classification job with arguments..." $@
uv run main_classification.py --gpu=1 $@