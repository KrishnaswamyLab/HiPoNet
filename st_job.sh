#!/bin/bash

#SBATCH --job-name=pcnet_st
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.long@yale.edu
#SBATCH --gpus=h200:1
#SBATCH --partition=gpu_h200

source .venv/bin/activate
uv run wandb login $WANDB_API_KEY
echo "Starting ST regression job with arguments..." $@
uv run main_ST.py --gpu=1 $@