#!/bin/bash
#SBATCH --job-name=pdo_multiscale_gen
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=./logs/slurm/pdo_variable_gen_%j.out
#SBATCH --error=./logs/slurm/pdo_variable_gen_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python

"$PY" train_variable_population_generator.py \
  --population_cache data/pdo_2m/populations_allcells_features44_n1678.npz \
  --latents checkpoints/legacy_gwt_k1_20260812/pdo2m/ld32_K1/train/latent_representations.npy \
  --output_dir checkpoints/pdo_multiscale_target_count_generator_20260814 \
  --validation_patient 75 --test_patient 99 \
  --count_steps 3000 --decoder_steps 5000 --flow_steps 4000 \
  --batch_populations 8 --count_batch_size 256 \
  --training_points 128 --training_point_choices 128,256,512 \
  --full_cloud_every 10 --full_cloud_moment_weight 0.05 \
  --metric_points 256 \
  --query_dim 64 --hidden_dim 256 --n_blocks 4 \
  --initial_scale 0.10 --maximum_scale 1.0 \
  --count_learning_rate 3e-4 --decoder_learning_rate 3e-4 \
  --flow_learning_rate 5e-5 --ot_epsilon 0.10 --ot_iterations 30 \
  --flow_distribution_weight 0.05 --velocity_weight 0.01 \
  --eval_every 200 --patience 1600 --integration_steps 32 \
  --eval_populations 48 --max_generated_cells 10000 \
  --seed 1818 --device cuda
