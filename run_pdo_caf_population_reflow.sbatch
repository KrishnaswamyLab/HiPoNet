#!/bin/bash
#SBATCH --job-name=pdo_caf_reflow
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=./logs/slurm/pdo_caf_reflow_%j.out
#SBATCH --error=./logs/slurm/pdo_caf_reflow_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python

"$PY" train_population_reflow.py \
  --population_cache data/pdo_caf_mendeley_all45_sampled512/populations_expanded_latest.npz \
  --latents checkpoints/pdo_caf_all45_hiponet_oneview_latent_K_sweep_20260730/ld32_K1/train/latent_representations.npy \
  --output_dir checkpoints/pdo_caf_sampled512_sinkhorn_no_context_reflow_20260817 \
  --validation_patient 75 --test_patient 99 \
  --decoder_steps 5000 --flow_steps 4000 \
  --noise_dim 16 --hidden_dim 256 --n_blocks 4 \
  --decoder_learning_rate 3e-4 --flow_learning_rate 5e-5 \
  --eval_every 200 --patience 1600 --integration_steps 32 \
  --metric_points 256 --eval_populations 0 --seed 1831 --device cuda
