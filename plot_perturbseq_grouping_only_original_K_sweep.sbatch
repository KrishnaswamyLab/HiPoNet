#!/bin/bash
#SBATCH --job-name=perturb_groupplot
#SBATCH --partition=day,scavenge
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-29%8
#SBATCH --output=./logs/slurm/perturb_groupplot_%A_%a.out
#SBATCH --error=./logs/slurm/perturb_groupplot_%A_%a.err

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
CACHE=perturb-seq/perturbseq_populations_gene_experiment_allcells_x_pca_50d.npz
ROOT=checkpoints/perturbseq_gene_experiment_grouping_only_original_training
TRAIN="$ROOT/ld${LD}_K${ORDER}/train"
OUT="$ROOT/phate_pngs"
mkdir -p "$OUT"

"$PY" utils/latent_space.py \
  --latents "$TRAIN/latent_representations.npy" \
  --labels "$TRAIN/labels.npy" \
  --population_cache "$CACHE" --color_by experiment \
  --knn "$KNN" --seed 1302 \
  --output "$OUT/perturbseq_grouping_only_original_K${ORDER}_ld${LD}_phate_knn${KNN}_by_experiment.png"
