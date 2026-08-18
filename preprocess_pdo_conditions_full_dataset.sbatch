#!/bin/bash
#SBATCH --job-name=pdo_conditions
#SBATCH --partition=scavenge
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --requeue
#SBATCH --output=./logs/slurm/pdo_conditions_%j.out
#SBATCH --error=./logs/slurm/pdo_conditions_%j.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm data/pdo_conditions_max_conc_no_O_H2O_all45_landmarks128

PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
ZIP=data/pdo_caf_mendeley/Metadata_PDO_CAF_screening.zip
LANDMARK_DIR=data/pdo_conditions_max_conc_no_O_H2O_all45_landmarks128
LANDMARK_CACHE="$LANDMARK_DIR/populations_expanded_latest.npz"
FULL_STORE=data/pdo_conditions_max_conc_no_O_H2O_all45_fullcells

"$PY" preprocess_pdo_caf_mendeley_full.py \
  --zip_path "$ZIP" --member Metadata_final_paper \
  --out_dir "$LANDMARK_DIR" --n_features 45 --min_cells_per_group 1 \
  --max_contexts 0 --landmark_cells_per_group 128 \
  --scaler_holdout_field Patient --scaler_holdout_values 75,99 \
  --pdo_conditions_only --seed 1860

"$PY" prepare_pdo_all45_fullcell_store.py \
  --zip_path "$ZIP" --member Metadata_final_paper \
  --population_cache "$LANDMARK_CACHE" \
  --out_dir "$FULL_STORE"

"$PY" -c "import json,numpy as np; p='$FULL_STORE'; x=np.load(p+'/cells.npy',mmap_mode='r'); o=np.load(p+'/population_offsets.npy'); c=np.load(p+'/condition_names.npy',allow_pickle=True); s=json.load(open(p+'/summary.json')); assert x.shape[1]==45 and len(o)==len(c)+1 and int(o[-1])==len(x) and s['n_populations']==len(c); print('validated PDO condition store',x.shape,'populations',len(c))"
