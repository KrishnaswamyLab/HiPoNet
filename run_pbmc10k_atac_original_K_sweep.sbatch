#!/bin/bash
#SBATCH --job-name=pbmc10k_K
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5%2
#SBATCH --output=./logs/slurm/pbmc10k_K_%A_%a.out
#SBATCH --error=./logs/slurm/pbmc10k_K_%A_%a.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm

DIMS=(16 32 50 16 32 50)
ORDERS=(1 1 1 2 2 2)
LD=${DIMS[$SLURM_ARRAY_TASK_ID]}
ORDER=${ORDERS[$SLURM_ARRAY_TASK_ID]}
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
CACHE=pbmc10k-atacdiff/pbmc10k_atac_cluster_pseudoreplicates_lsi50.npz
DIST=checkpoints/pbmc10k_atac_pheno_gs_shared_k10/pheno_gs_distances.npy
ROOT=checkpoints/pbmc10k_atac_grouping_original_K_sweep
TRAIN="$ROOT/ld${LD}_K${ORDER}/train"
mkdir -p "$TRAIN"

"$PY" -c "import numpy as np; d=np.load('$CACHE',allow_pickle=True); p=d['populations']; assert len(p)==152; assert sum(len(x) for x in p)==11909; assert all(x.shape[1]==50 for x in p); q=np.load('$DIST'); assert q.shape==(152,152) and np.isfinite(q).all() and np.allclose(q,q.T); print('validated PBMC10k cache and Pheno-GS matrix')"

GEOMETRIC_ARGS=()
if [[ "$ORDER" -eq 2 ]]; then
  GEOMETRIC_ARGS=(
    --use_geometric_laplacian
    --simplicial_max_neighbors 12
    --simplicial_max_triangles 250
  )
fi

"$PY" unsupervised_main.py \
  --gpu 0 \
  --raw_dir population_npz --population_cache "$CACHE" \
  --num_epochs 120 --batch_size 4 --n_accumulate 2 \
  --latent_dim "$LD" --hidden_dims 256 128 --num_weights 1 \
  --K "$ORDER" --J 2 "${GEOMETRIC_ARGS[@]}" \
  --sigma 32 --threshold 0.1 --ignore_alphas \
  --dist_weight 0.1 --udemd_cache "$DIST" \
  --val_fraction 0.2 --seed 1302 --split_seed 1302 \
  --disable_wb --skip_phate --save_dir "$TRAIN"

"$PY" -c "import numpy as np; z=np.load('$TRAIN/latent_representations.npy'); assert z.shape==(152,$LD) and np.isfinite(z).all(); print('validated PBMC10k latent export',z.shape,'K=$ORDER')"
