#!/bin/bash
#SBATCH --job-name=bms_allprep
#SBATCH --partition=day,scavenge
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=./logs/slurm/bms_allprep_%j.out
#SBATCH --error=./logs/slurm/bms_allprep_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm data/pheno_gs_bms checkpoints/bms_allcells_pheno_gs_t10_k5

PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
ZIP=data/pheno_gs_bms/Pheno-GS_BMS.zip
CACHE=data/pheno_gs_bms/populations_allcells_features34_n75.npz
DISTDIR=checkpoints/bms_allcells_pheno_gs_t10_k5

if [[ ! -s "$CACHE" ]]; then
  "$PY" preprocess_bms_allcells.py --zip_path "$ZIP" --output "$CACHE"
fi

if [[ ! -s "$DISTDIR/pheno_gs_distances.npy" ]]; then
  "$PY" prepare_pheno_gs_distances.py \
    --cache "$CACHE" --out_dir "$DISTDIR" \
    --diffusion_time 10 --chebyshev_order 5 --knn 10 --n_jobs 8
fi

"$PY" -c "import numpy as np; c=np.load('$CACHE',allow_pickle=True); p=list(c['populations']); assert len(p)==75 and sum(map(len,p))==60693 and {x.shape[1] for x in p}=={34}; assert int(c['landmark_cells_per_group'][0])==-1; d=np.load('$DISTDIR/pheno_gs_distances.npy'); assert d.shape==(75,75) and np.isfinite(d).all() and np.allclose(d,d.T); print('validated BMS all-cell cache and Pheno-GS', [len(x) for x in p][:5])"
