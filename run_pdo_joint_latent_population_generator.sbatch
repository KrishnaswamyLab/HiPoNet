#!/bin/bash
#SBATCH --job-name=pdo_joint_gen
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/pdo_joint_gen_%j.out
#SBATCH --error=./logs/slurm/pdo_joint_gen_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python

"$PY" finetune_latent_population_generator.py \
  --population_cache data/pdo_2m/populations_allcells_features44_n1678.npz \
  --latents checkpoints/legacy_gwt_k1_20260812/pdo2m/ld32_K1/train/latent_representations.npy \
  --checkpoint checkpoints/pdo_multiscale_target_count_generator_20260814/variable_population_generator.pt \
  --output_dir checkpoints/pdo_joint_latent_population_generator_20260817 \
  --validation_patient 75 --test_patient 99 \
  --steps 2500 --training_points 128,256,512 --cell_budget 1024 \
  --query_dim 64 --hidden_dim 256 --n_blocks 4 \
  --learning_rate 2e-5 --source_weight 0.10 \
  --flow_matching_weight 0.10 --full_cloud_weight 0.05 \
  --full_cloud_every 25 --ot_epsilon 0.10 --ot_iterations 30 \
  --train_integration_steps 4 --integration_steps 32 \
  --eval_every 100 --patience 1000 --metric_points 256 \
  --eval_populations 48 --seed 1820 --device cuda
