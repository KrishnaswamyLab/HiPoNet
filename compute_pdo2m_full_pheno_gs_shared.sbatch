#!/bin/bash
#SBATCH --job-name=pdo2m_phgs_cols
#SBATCH --partition=day,scavenge
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-127%16
#SBATCH --output=./logs/slurm/pdo2m_phgs_cols_%A_%a.out
#SBATCH --error=./logs/slurm/pdo2m_phgs_cols_%A_%a.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
"$PY" prepare_full_pheno_gs_shared.py compute \
  --cache data/pdo_2m/populations_allcells_features44_n1678.npz \
  --out_dir checkpoints/pdo2m_full_pheno_gs_shared_k10_pynn \
  --diffusion_time 10 --chebyshev_order 5 \
  --n_chunks 128 --chunk_id "$SLURM_ARRAY_TASK_ID"
