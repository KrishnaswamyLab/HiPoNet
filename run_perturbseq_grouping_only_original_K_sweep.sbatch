#!/bin/bash
#SBATCH --job-name=perturb_groupbase
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5%2
#SBATCH --output=./logs/slurm/perturb_groupbase_%A_%a.out
#SBATCH --error=./logs/slurm/perturb_groupbase_%A_%a.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm

DIMS=(16 32 50 16 32 50)
ORDERS=(1 1 1 2 2 2)
LD=${DIMS[$SLURM_ARRAY_TASK_ID]}
ORDER=${ORDERS[$SLURM_ARRAY_TASK_ID]}
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
CACHE=perturb-seq/perturbseq_populations_gene_experiment_allcells_x_pca_50d.npz
DIST=checkpoints/perturbseq_gene_experiment_pheno_gs_shared_k10_pynn/pheno_gs_distances.npy
ROOT=checkpoints/perturbseq_gene_experiment_grouping_only_original_training
TRAIN="$ROOT/ld${LD}_K${ORDER}/train"
mkdir -p "$TRAIN"

"$PY" -c "import numpy as np; d=np.load('$CACHE',allow_pickle=True); p=d['populations']; assert len(p)==5164; assert sum(len(x) for x in p)==214449; assert all(x.shape[1]==50 for x in p); assert set(d['experiment'].astype(str))=={'control','treatment'}; q=np.load('$DIST'); assert q.shape==(5164,5164) and np.isfinite(q).all() and np.allclose(q,q.T); print('validated grouping-only cache and Pheno-GS matrix')"

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

"$PY" -c "import numpy as np; z=np.load('$TRAIN/latent_representations.npy'); assert z.shape==(5164,$LD) and np.isfinite(z).all(); print('validated grouping-only latent export',z.shape,'K=$ORDER')"
