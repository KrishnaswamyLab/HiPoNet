#!/bin/bash
#SBATCH --job-name=mendeley_phate
#SBATCH --partition=day,scavenge
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=0-39%8
#SBATCH --output=slurm-mendeley-phate-%A_%a.out
#SBATCH --error=slurm-mendeley-phate-%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
cd "$PROJECT_DIR"

DIMS=(16 32 45 64 16 32 45 64)
ORDERS=(1 1 1 1 2 2 2 2)
KNN_VALUES=(5 10 20 30 50)

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
KNN_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
LATENT_DIM=${DIMS[$MODEL_INDEX]}
ORDER=${ORDERS[$MODEL_INDEX]}
PHATE_KNN=${KNN_VALUES[$KNN_INDEX]}

if [[ -z ${PYTHON:-} ]]; then
  if [[ -x "$PROJECT_DIR/.vendor/phate-env/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.vendor/phate-env/bin/python"
  else
    PYTHON=python
  fi
fi
CACHE=${CACHE:-data/pdo_caf_mendeley/populations_latest.npz}
ROOT=${ROOT:-checkpoints/mendeley_hiponet_oneview_latent_K_sweep}
TRAIN_DIR="$ROOT/ld${LATENT_DIM}_K${ORDER}/train"
LATENTS="$TRAIN_DIR/latent_representations.npy"
OUTPUT_DIR="$ROOT/pngs"
OUTPUT="$OUTPUT_DIR/mendeley_oneview_phate_ld${LATENT_DIM}_K${ORDER}_knn${PHATE_KNN}.png"

test -s "$CACHE"
test -s "$LATENTS"
mkdir -p "$OUTPUT_DIR"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/cache-${SLURM_JOB_ID}"

"$PYTHON" plot_mendeley_phate_sweep.py \
  --latents "$LATENTS" \
  --cache "$CACHE" \
  --output "$OUTPUT" \
  --phate_knn "$PHATE_KNN" \
  --simplicial_k "$ORDER" \
  --latent_dim "$LATENT_DIM" \
  --seed 1302 \
  --n_jobs "$SLURM_CPUS_PER_TASK"

test -s "$OUTPUT"
