#!/bin/bash
#SBATCH --job-name=bms_allphate
#SBATCH --partition=day,scavenge
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-39%8
#SBATCH --output=./logs/slurm/bms_allphate_%A_%a.out
#SBATCH --error=./logs/slurm/bms_allphate_%A_%a.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm

DIMS=(16 32 45 64 16 32 45 64)
ORDERS=(1 1 1 1 2 2 2 2)
KNN_VALUES=(5 10 20 30 50)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
KNN_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
LD=${DIMS[$MODEL_INDEX]}
ORDER=${ORDERS[$MODEL_INDEX]}
KNN=${KNN_VALUES[$KNN_INDEX]}
ROOT=checkpoints/bms_allcells_hiponet_oneview_latent_K_sweep_20260803
CACHE=data/pheno_gs_bms/populations_allcells_features34_n75.npz
LATENTS="$ROOT/ld${LD}_K${ORDER}/train/latent_representations.npy"
OUT_PNG="$ROOT/pngs/hiponet_bms_allcells_oneview_phate_ld${LD}_K${ORDER}_knn${KNN}_treatment_patient_microenvironment.png"
ARTIFACTS="$ROOT/phate_artifacts/ld${LD}_K${ORDER}_knn${KNN}"

test -s "$LATENTS"
mkdir -p "$ROOT/pngs"
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}

"$PWD/.vendor/phate-env/bin/python" plot_bms_phate_sweep.py \
  --latents "$LATENTS" --cache "$CACHE" --output "$OUT_PNG" \
  --artifact_dir "$ARTIFACTS" --phate_knn "$KNN" \
  --simplicial_k "$ORDER" --latent_dim "$LD" --seed 1302
test -s "$OUT_PNG"
