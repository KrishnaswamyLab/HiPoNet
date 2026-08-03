# DDR Pre-Analysis (for supervised HiPoNet)

This folder contains a pre-analysis pipeline for DDR data before supervised HiPoNet training.

## Files

- `pre_analysis.py`: builds per-cell engineered features from DDR parquet tables, runs QC, and saves PCA/PHATE/UMAP/t-SNE plots.
- `output/`: default output folder for generated tables and figures.

## Expected input

A directory with at least:

- a cells table matching `cells_classified*.parquet`
- a foci table matching one of:
  - `*MultiFoci*.parquet`
  - `*NuclearFoci*.parquet`
  - `*Foci*.parquet`

Default input path is:

`/nfs/roberts/pi/pi_sk2433/shared/JohnLock_2026_DDR`

## Install dependencies

This script needs parquet support and plotting libraries:

```bash
pip install pandas pyarrow matplotlib scikit-learn
```

Optional (for extra embeddings):

```bash
pip install phate umap-learn
```

## Run

From project root:

```bash
python3 DDR/pre_analysis.py \
  --data_dir /nfs/roberts/pi/pi_sk2433/shared/JohnLock_2026_DDR \
  --out_dir DDR/output \
  --max_cells 80000 \
  --max_foci_rows 2500000
```

## Main outputs

- `schema_summary.json`: shapes and columns from cells/foci tables
- `feature_build_summary.json`: inferred key columns and feature build stats
- `missingness.csv`: missing-value fractions per column
- `label_balance.csv`: class distribution if label was inferred
- `cell_level_features.parquet`: engineered per-cell features
- `embedding_*.csv`: 2D coordinates
- `embedding_*.png`: PCA/PHATE/UMAP/t-SNE scatter plots
- `embedding_run_log.json`: which embedding methods ran or were skipped

## Notes

- If PHATE/UMAP are not installed, the script still runs PCA and usually t-SNE.
- Column inference uses common naming patterns (cell id, label, marker, batch). If inference fails, update candidate lists at the top of `pre_analysis.py`.
