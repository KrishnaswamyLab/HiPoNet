#!/bin/bash
#SBATCH --job-name=pdo2m_pairgs
#SBATCH --partition=day,scavenge
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/pdo2m_pairgs_%j.out
#SBATCH --error=./logs/slurm/pdo2m_pairgs_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm checkpoints/pdo2m_pairwise_pheno_trial_n16
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python

"$PY" prepare_pairwise_pheno_gs_trial.py \
  --cache data/pdo_2m/populations_allcells_features44_n1678.npz \
  --out_dir checkpoints/pdo2m_pairwise_pheno_trial_n16 \
  --n_populations 16 --seed 1302 --knn 10 \
  --sigma 32 --threshold 0.1 \
  --diffusion_time 10 --chebyshev_order 5 --n_jobs 4

"$PY" -c "import numpy as np; d=np.load('checkpoints/pdo2m_pairwise_pheno_trial_n16/pheno_gs_distances.npy'); assert d.shape==(16,16) and np.isfinite(d).all() and np.allclose(d,d.T); print('validated pairwise Pheno-GS trial',d.shape)"
