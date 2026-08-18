#!/bin/bash
#SBATCH --job-name=pdo_emd_reflow
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/pdo_emd_reflow_%j.out
#SBATCH --error=./logs/slurm/pdo_emd_reflow_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python

"$PY" train_population_reflow.py \
  --population_cache data/pdo_2m/populations_allcells_features44_n1678.npz \
  --latents checkpoints/legacy_gwt_k1_20260812/pdo2m/ld32_K1/train/latent_representations.npy \
  --output_dir checkpoints/pdo_emd_only_soft_reflow_20260818 \
  --validation_patient 75 --test_patient 99 \
  --decoder_steps 5000 --flow_steps 4000 \
  --noise_dim 16 --hidden_dim 256 --n_blocks 4 \
  --decoder_learning_rate 3e-4 --flow_learning_rate 5e-5 \
  --emd_solver sinkhorn_log --emd_regularization 0.2 --emd_iterations 100 \
  --eval_every 200 --patience 1600 --integration_steps 32 \
  --metric_points 256 --eval_populations 0 --seed 1870 --device cuda
