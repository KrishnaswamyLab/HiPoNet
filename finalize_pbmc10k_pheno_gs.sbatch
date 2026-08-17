#!/bin/bash
#SBATCH --job-name=pbmc10k_join
#SBATCH --partition=day,scavenge
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=./logs/slurm/pbmc10k_join_%j.out
#SBATCH --error=./logs/slurm/pbmc10k_join_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
CACHE=pbmc10k-atacdiff/pbmc10k_atac_cluster_pseudoreplicates_lsi50.npz
OUT=checkpoints/pbmc10k_atac_pheno_gs_shared_k10

"$PY" prepare_full_pheno_gs_shared.py finalize \
  --cache "$CACHE" --out_dir "$OUT" --n_chunks 32
