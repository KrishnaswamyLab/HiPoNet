#!/bin/bash

#SBATCH --job-name=pcnet_tr
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.long@yale.edu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --partition=gpu

source .venv/bin/activate
uv run wandb login $WANDB_API_KEY
echo "Starting classification job with arguments..." $@
uv run main_transformer_classification.py --gpu=1 $@