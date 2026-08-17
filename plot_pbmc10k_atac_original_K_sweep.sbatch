#!/bin/bash
#SBATCH --job-name=pbmc10k_phate
#SBATCH --partition=day,scavenge
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-29%8
#SBATCH --output=./logs/slurm/pbmc10k_phate_%A_%a.out
#SBATCH --error=./logs/slurm/pbmc10k_phate_%A_%a.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm

DIMS=(16 32 50 16 32 50)
ORDERS=(1 1 1 2 2 2)
KNNS=(5 10 20 30 50)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
KNN_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
LD=${DIMS[$MODEL_INDEX]}
ORDER=${ORDERS[$MODEL_INDEX]}
KNN=${KNNS[$KNN_INDEX]}
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
CACHE=pbmc10k-atacdiff/pbmc10k_atac_cluster_pseudoreplicates_lsi50.npz
ROOT=checkpoints/pbmc10k_atac_grouping_original_K_sweep
TRAIN="$ROOT/ld${LD}_K${ORDER}/train"
OUT="$ROOT/phate_pngs"
mkdir -p "$OUT"

"$PY" utils/latent_space.py \
  --latents "$TRAIN/latent_representations.npy" \
  --labels "$TRAIN/labels.npy" \
  --population_cache "$CACHE" --color_by atac_cluster \
  --knn "$KNN" --seed 1302 \
  --output "$OUT/pbmc10k_atac_original_K${ORDER}_ld${LD}_phate_knn${KNN}_by_cluster.png"
