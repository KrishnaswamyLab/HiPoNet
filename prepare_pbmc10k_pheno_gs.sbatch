#!/bin/bash
#SBATCH --job-name=pbmc10k_graph
#SBATCH --partition=day,scavenge
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=./logs/slurm/pbmc10k_graph_%j.out
#SBATCH --error=./logs/slurm/pbmc10k_graph_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
CACHE=pbmc10k-atacdiff/pbmc10k_atac_cluster_pseudoreplicates_lsi50.npz
OUT=checkpoints/pbmc10k_atac_pheno_gs_shared_k10

"$PY" prepare_full_pheno_gs_shared.py prepare \
  --cache "$CACHE" --out_dir "$OUT" \
  --cell_knn 10 --search_neighbors 32 --decay 40 \
  --kernel_threshold 1e-4 --diffusion_time 10 --chebyshev_order 5 \
  --n_chunks 32 --n_jobs "$SLURM_CPUS_PER_TASK" --seed 1302
