#!/bin/bash
#SBATCH --job-name=pbmc10k_lsi
#SBATCH --partition=day,scavenge
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/pbmc10k_lsi_%j.out
#SBATCH --error=./logs/slurm/pbmc10k_lsi_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
BASE=pbmc10k-atacdiff
CACHE="$BASE/pbmc10k_atac_cluster_pseudoreplicates_lsi50.npz"

"$PY" prepare_pbmc10k_atac_populations.py \
  --matrix "$BASE/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5" \
  --clusters "$BASE/analysis/clustering/atac/graphclust/clusters.csv" \
  --output "$CACHE" --n_components 50 --replicates_per_cluster 8 --seed 1302

"$PY" -c "import numpy as np; d=np.load('$CACHE',allow_pickle=True); p=d['populations']; assert len(p)==152; assert sum(len(x) for x in p)==11909; assert all(x.shape[1]==50 for x in p); assert len(np.unique(d['atac_cluster']))==19; print('validated PBMC10k all-cell population cache')"
