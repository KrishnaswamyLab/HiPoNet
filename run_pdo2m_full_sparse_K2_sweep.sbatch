#!/bin/bash
#SBATCH --job-name=pdo2m_full_K2s
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-3%2
#SBATCH --output=./logs/slurm/pdo2m_full_K2s_%A_%a.out
#SBATCH --error=./logs/slurm/pdo2m_full_K2s_%A_%a.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm

DIMS=(16 32 45 64)
LD=${DIMS[$SLURM_ARRAY_TASK_ID]}
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
CACHE=data/pdo_2m/populations_allcells_features44_n1678.npz
ROOT=checkpoints/pdo2m_full_1678_hiponet_sparse_K2_sweep
DIST=checkpoints/pdo2m_full_pheno_gs_shared_k10_pynn/pheno_gs_distances.npy
TRAIN="$ROOT/ld${LD}_K2/train"
mkdir -p "$TRAIN"

"$PY" -c "import numpy as np; d=np.load('$CACHE',allow_pickle=True); p=d['populations']; assert len(p)==1678; assert sum(len(x) for x in p)==2000000; assert all(x.shape[1]==44 for x in p); print('validated full cache: 1678 populations, 2000000 cells, 44 features')"

"$PY" unsupervised_main.py \
  --gpu 0 \
  --raw_dir population_npz --population_cache "$CACHE" \
  --num_epochs 120 --batch_size 4 --n_accumulate 2 \
  --latent_dim "$LD" --hidden_dims 256 128 --num_weights 1 --K 2 --J 2 \
  --use_geometric_laplacian --simplicial_max_neighbors 12 \
  --simplicial_max_triangles 250 \
  --sigma 32 --threshold 0.1 --ignore_alphas \
  --dist_weight 0.1 --udemd_cache "$DIST" \
  --split_cache "$CACHE" --split_field Patient --val_group 75 --test_group 99 \
  --seed 1302 --split_seed 1302 --disable_wb --save_dir "$TRAIN"

"$PY" -c "import numpy as np; z=np.load('$TRAIN/latent_representations.npy'); assert z.shape==(1678,$LD) and np.isfinite(z).all(); print('validated full sparse K2 latent export',z.shape)"
