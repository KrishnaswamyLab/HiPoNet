#!/bin/bash
#SBATCH --job-name=pdo2m_phgs_graph
#SBATCH --partition=day,scavenge
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=512G
#SBATCH --output=./logs/slurm/pdo2m_phgs_graph_%j.out
#SBATCH --error=./logs/slurm/pdo2m_phgs_graph_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
"$PY" prepare_full_pheno_gs_shared.py prepare \
  --cache data/pdo_2m/populations_allcells_features44_n1678.npz \
  --out_dir checkpoints/pdo2m_full_pheno_gs_shared_k10_pynn \
  --cell_knn 10 --search_neighbors 32 --decay 40 --kernel_threshold 1e-4 \
  --diffusion_time 10 --chebyshev_order 5 \
  --n_chunks 128 --n_jobs 24 --seed 1302
