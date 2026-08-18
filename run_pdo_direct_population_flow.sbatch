#!/bin/bash
#SBATCH --job-name=pdo_direct_fm
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/pdo_direct_fm_%j.out
#SBATCH --error=./logs/slurm/pdo_direct_fm_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python

"$PY" train_direct_population_flow.py \
  --population_cache data/pdo_2m/populations_allcells_features44_n1678.npz \
  --latents checkpoints/legacy_gwt_k1_20260812/pdo2m/ld32_K1/train/latent_representations.npy \
  --output_dir checkpoints/pdo_direct_gaussian_cfm_z_conditioned_20260817 \
  --validation_patient 75 --test_patient 99 \
  --steps 5000 --hidden_dim 256 --n_blocks 4 --learning_rate 1e-4 \
  --eval_every 200 --patience 1600 --integration_steps 32 \
  --metric_points 256 --eval_populations 0 --seed 1840 --device cuda
