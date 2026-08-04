#!/bin/bash

#SBATCH --job-name=hiponet_perturbseq_preproc
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=512G
#SBATCH --output=slurm-perturbseq-preproc-%j.out
#SBATCH --error=slurm-perturbseq-preproc-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

source .venv/bin/activate

export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-$$}"
export XDG_CACHE_HOME="/tmp/cache-${SLURM_JOB_ID:-$$}"

mkdir -p checkpoints/perturbseq_preprocessed_pca50

echo "Starting unsupervised perturb-seq HiPoNet job (preprocessed PCA cache)"
echo "Working directory: $PWD"

python unsupervised_main.py \
  --raw_dir population_npz \
  --population_cache perturb-seq/perturbseq_populations_preprocessed_x_pca_50d.pkl \
  --save_dir checkpoints/perturbseq_preprocessed_pca50 \
  --latent_dim 32 \
  --num_epochs 80 \
  --batch_size 2 \
  --dist_weight 0 \
  --phate_color_by experiment \
  --disable_wb