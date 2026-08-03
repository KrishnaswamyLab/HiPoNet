#!/bin/bash
#SBATCH --job-name=bms_allK1
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-3%3
#SBATCH --output=./logs/slurm/bms_allK1_%A_%a.out
#SBATCH --error=./logs/slurm/bms_allK1_%A_%a.err

set -euo pipefail
cd /nfs/roberts/project/pi_sk2433/edc26/HiPoNet
mkdir -p logs/slurm

DIMS=(16 32 45 64)
LD=${DIMS[$SLURM_ARRAY_TASK_ID]}
PY=/nfs/roberts/project/pi_sk2433/edc26/.venv/bin/python
CACHE=data/pheno_gs_bms/populations_allcells_features34_n75.npz
DIST=checkpoints/bms_allcells_pheno_gs_t10_k5/pheno_gs_distances.npy
ROOT=checkpoints/bms_allcells_hiponet_oneview_latent_K_sweep_20260803
TRAIN="$ROOT/ld${LD}_K1/train"

if [[ -s "$TRAIN/latent_representations.npy" ]]; then exit 0; fi

"$PY" unsupervised_main.py \
  --gpu 0 --autoencoder_target wavelet --generator_weight 0 \
  --raw_dir population_npz --population_cache "$CACHE" \
  --num_epochs 120 --batch_size 2 --n_accumulate 4 \
  --latent_dim "$LD" --hidden_dims 256 128 \
  --num_weights 1 --K 1 --J 2 \
  --dist_weight 0.1 --udemd_cache "$DIST" --val_fraction 0.2 \
  --seed 1302 --split_seed 1302 --disable_wb --save_dir "$TRAIN"

"$PY" evaluate_latent_geosink.py \
  --latents "$TRAIN/latent_representations.npy" --distances "$DIST" \
  --training_summary "$TRAIN/training_summary.json" \
  --out "$TRAIN/latent_pheno_gs_metrics.json"
"$PY" -c "import numpy as np; z=np.load('$TRAIN/latent_representations.npy'); assert z.shape==(75,$LD) and np.isfinite(z).all(); print('validated BMS all-cell one-view',z.shape,'K=1')"
