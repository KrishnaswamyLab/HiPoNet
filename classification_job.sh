#!/bin/bash

#SBATCH --job-name=hiponet_classification
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu

source .venv/bin/activate
echo "Starting classification job with arguments..." $@
uv run main_classification.py --gpu=1 $@