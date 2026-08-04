#!/bin/bash
#SBATCH --job-name=pdo2m_pairhipo
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=./logs/slurm/pdo2m_pairhipo_%j.out
#SBATCH --error=./logs/slurm/pdo2m_pairhipo_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm checkpoints/pdo2m_pairwise_hiponet_trial_n16
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
PHATE_PY="$PWD/.vendor/phate-env/bin/python"
CACHE=checkpoints/pdo2m_pairwise_pheno_trial_n16/population_subset.npz
DIST=checkpoints/pdo2m_pairwise_pheno_trial_n16/pheno_gs_distances.npy
OUT=checkpoints/pdo2m_pairwise_hiponet_trial_n16

AUTOENCODER_ARGS=()
if "$PY" unsupervised_main.py --help | grep -q -- "--autoencoder_target"; then
  AUTOENCODER_ARGS=(--autoencoder_target wavelet --generator_weight 0)
fi
PHATE_ARGS=()
if "$PY" unsupervised_main.py --help | grep -q -- "--skip_phate"; then
  PHATE_ARGS=(--skip_phate)
fi

"$PY" unsupervised_main.py "${AUTOENCODER_ARGS[@]}" "${PHATE_ARGS[@]}" \
  --gpu 0 \
  --raw_dir population_npz --population_cache "$CACHE" \
  --num_epochs 20 --batch_size 2 --n_accumulate 2 \
  --latent_dim 16 --hidden_dims 128 64 --num_weights 1 --K 1 --J 2 \
  --sigma 32 --threshold 0.1 --ignore_alphas \
  --dist_weight 0.1 --udemd_cache "$DIST" \
  --seed 1302 --split_seed 1302 --disable_wb --save_dir "$OUT"

"$PHATE_PY" utils/latent_space.py \
  --latents "$OUT/latent_representations.npy" \
  --labels "$OUT/labels.npy" --population_cache "$CACHE" \
  --color_by Treatment --knn 5 --seed 1302 \
  --output "$OUT/pdo2m_pairwise_pheno_trial_phate_treatment.png"
