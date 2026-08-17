#!/bin/bash
#SBATCH --job-name=pbmc10k_wave
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=./logs/slurm/pbmc10k_wave_%j.out
#SBATCH --error=./logs/slurm/pbmc10k_wave_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
CACHE=pbmc10k-atacdiff/pbmc10k_atac_cluster_pseudoreplicates_lsi50.npz
OUT=checkpoints/pbmc10k_atac_cached_wavelets/K2_wavelet_features.npy

"$PY" cache_perturbseq_wavelet_features.py \
  --cache "$CACHE" --output "$OUT" --K 2 --batch_size 4 --device cuda
