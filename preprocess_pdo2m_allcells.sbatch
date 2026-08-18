#!/bin/bash
#SBATCH --job-name=pdo2m_cache
#SBATCH --partition=day,scavenge
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=./logs/slurm/pdo2m_cache_%j.out
#SBATCH --error=./logs/slurm/pdo2m_cache_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm data/pdo_2m
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
H5AD=data/pdo_2m/2m_pdo.h5ad
CACHE=data/pdo_2m/populations_allcells_features44_n1678.npz

if [[ ! -s "$CACHE" ]]; then
  "$PY" preprocess_pdo2m_allcells.py --h5ad "$H5AD" --output "$CACHE"
fi
"$PY" -c "import numpy as np; c=np.load('$CACHE',allow_pickle=True); p=list(c['populations']); assert len(p)==1678 and sum(map(len,p))==2000000 and {x.shape[1] for x in p}=={44}; assert int(c['landmark_cells_per_group'][0])==-1; print('validated PDO2M all-cell cache',len(p),sum(map(len,p)),min(map(len,p)),max(map(len,p)))"
