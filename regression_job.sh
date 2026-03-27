#!/bin/bash

#SBATCH --job-name=pcnet_reg
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --output=./logs/slurm/regression/melanoma/%x_%j.out
#SBATCH --error=./logs/slurm/regression/melanoma/%x_%j.err

source .venv/bin/activate
echo "Starting regression job with arguments..." $@
uv run main_regression.py --gpu=0 $@
