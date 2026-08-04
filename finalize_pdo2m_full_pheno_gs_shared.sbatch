#!/bin/bash
#SBATCH --job-name=pdo2m_phgs_join
#SBATCH --partition=day,scavenge
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=./logs/slurm/pdo2m_phgs_join_%j.out
#SBATCH --error=./logs/slurm/pdo2m_phgs_join_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
"$PY" prepare_full_pheno_gs_shared.py finalize \
  --cache data/pdo_2m/populations_allcells_features44_n1678.npz \
  --out_dir checkpoints/pdo2m_full_pheno_gs_shared_k10_pynn \
  --n_chunks 128
