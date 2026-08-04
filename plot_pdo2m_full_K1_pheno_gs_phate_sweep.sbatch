#!/bin/bash
#SBATCH --job-name=pdo2m_K1pg_phate
#SBATCH --partition=day,scavenge
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=0-19%8
#SBATCH --output=./logs/slurm/pdo2m_K1pg_phate_%A_%a.out
#SBATCH --error=./logs/slurm/pdo2m_K1pg_phate_%A_%a.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet

DIMS=(16 32 45 64)
KNN_VALUES=(5 10 20 30 50)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
KNN_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
LD=${DIMS[$MODEL_INDEX]}
KNN=${KNN_VALUES[$KNN_INDEX]}
ROOT=checkpoints/pdo2m_full_1678_hiponet_pheno_gs_K1_sweep
CACHE=data/pdo_2m/populations_allcells_features44_n1678.npz
LATENTS="$ROOT/ld${LD}_K1/train/latent_representations.npy"
OUT="$ROOT/pngs/hiponet_pdo2m_full1678_phate_ld${LD}_K1_knn${KNN}.png"
ART="$ROOT/phate_artifacts/ld${LD}_K1_knn${KNN}"

test -s "$LATENTS"
mkdir -p "$ROOT/pngs"
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
"$PWD/.vendor/phate-env/bin/python" plot_pdo2m_phate_sweep.py \
  --latents "$LATENTS" --cache "$CACHE" --output "$OUT" \
  --artifact_dir "$ART" --phate_knn "$KNN" \
  --simplicial_k 1 --latent_dim "$LD" --seed 1302
test -s "$OUT"
