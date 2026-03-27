# Changes — Geometric Hodge Laplacian & Experiments

**Commit:** `dd9558a` — *Add geom laplacian and experiments and tests*
**Branch:** `materials_gen`

---

## Summary

This commit introduces the **geometric (metric-aware) Hodge Laplacian** as an alternative to the combinatorial Hodge Laplacian in HiPoNet's simplicial wavelet transform pipeline. It also adds five standalone experiments validating the geometric Laplacian, a new attention pooling mechanism, adaptive threshold selection, sparsity regularization, and a comprehensive test suite.

---

## New Modules

### 1. `models/hodge_laplacian.py` — Hodge Laplacian Math

Both numpy (for standalone use/experiments) and **pure-torch** (for training, fully differentiable) implementations of the geometric Hodge Laplacian:

```
Δ_k = M_k⁻¹ B_{k+1} M_{k+1} B_{k+1}^T  +  B_k^T M_{k-1}⁻¹ B_k M_k
```

Key functions:
- `cayley_menger_volumes(simplex_list, sq_dist_matrix)` — compute k-simplex volumes from pairwise squared distances via Cayley-Menger determinants (torch, differentiable)
- `geometric_hodge_laplacian_torch(B_k, B_kp1, v_km1, v_k, v_kp1)` — assemble the full geometric Hodge k-Laplacian in pure torch
- `compute_simplex_volumes(sc, points)` — numpy version for experiments
- `geometric_hodge_laplacian(sc, volumes, k)` — sparse numpy version for experiments

### 2. `models/simplicial_complex.py` — `SimplicialComplex` Class

Vietoris-Rips simplicial complex builder with properly oriented boundary matrices (d²=0).

```python
from models.simplicial_complex import SimplicialComplex

sc = SimplicialComplex(points, max_dim=2, threshold=0.5)
B1 = sc.boundary_matrix(1)  # vertices → edges
B2 = sc.boundary_matrix(2)  # edges → triangles
```

### 3. `models/diffusion_maps.py` — Diffusion Maps Embedding

Standalone diffusion maps implementation via `scipy.sparse.linalg.eigsh`. Used in experiments (not in the training pipeline).

```python
from models.diffusion_maps import diffusion_maps

coords = diffusion_maps(X, n_components=10, epsilon=None, alpha=1.0)
```

### 4. `models/threshold_selection.py` — Adaptive Threshold via Kneedle

Automatically selects the graph construction threshold by finding the knee in the edge-count vs. threshold curve.

```python
from models.threshold_selection import select_threshold

threshold = select_threshold(adj_matrix, n_points=50, sigma=0.5)
```

Usage in training: pass `--use_kneedle` to `main_regression.py` to auto-select threshold per fold.

---

## Modified Modules

### 5. `models/SWT.py` — `SimplicialWaveletTransform` (Major Rewrite)

**New features:**
- **Geometric Laplacian mode:** Pass `use_geometric_laplacian=True` and `sq_diff_dists` (squared diffusion distances) to use metric-aware transition matrices instead of combinatorial ones. Fully differentiable — no scipy, no eigendecomposition.
- **`PoolingAttention`** — DeepSet attention pooling over simplices. Learns per-simplex attention weights for interpretable aggregation. Weights accessible via `self.last_weights` after each forward pass.

```python
swt = SimplicialWaveletTransform(
    adj, features, threshold, device,
    use_geometric_laplacian=True,
    sq_diff_dists=sq_dists_tensor,
)
coeffs = swt.calculate_wavelet_coeff(J=3)
```

### 6. `models/graph_learning.py` — New `SimplicialFeatLearningLayer` + HiPoNet Updates

**New class: `SimplicialFeatLearningLayer`** — unified simplicial feature learning layer replacing the older `SimplicialFeatLearningLayerTri`/`Tetra`. Supports:
- Combinatorial or geometric Hodge Laplacian (`use_geometric_laplacian`)
- Configurable diffusion steps (`diffusion_steps`)
- Optional attention pooling (`use_attention`)
- Separate node features for spatial views (`node_features` parameter)

**`GraphFeatLearningLayer`** now also accepts optional `node_features` in `forward()`.

**`HiPoNet`** constructor updated to pass through the new flags.

### 7. `main_classification.py` — New CLI Arguments

```bash
uv run python main_classification.py \
    --K 2 \
    --use_geometric_laplacian \
    --diffusion_steps 3 \
    --use_attention \
    --sparse --sparse_lambda 0.01
```

| Flag | Description |
|------|-------------|
| `--use_geometric_laplacian` | Use metric-aware geometric Hodge Laplacian (requires `K >= 2`) |
| `--diffusion_steps N` | Number of diffusion steps for computing P^t (default: 1) |
| `--use_attention` | Use DeepSet attention pooling over simplices (K >= 2) |
| `--sparse` | Add L1 sparsity loss on learnable alphas |
| `--sparse_lambda` | Weight for the L1 sparsity loss (default: 0.01) |

### 8. `main_regression.py` — Major Update

- Switched from train/test split to **K-Fold cross-validation** (`sklearn.model_selection.KFold`)
- Added all the same new flags as classification (`--use_geometric_laplacian`, `--diffusion_steps`, `--sparse`, etc.)
- Added `--use_kneedle` for adaptive threshold selection per fold
- Added `--K`, `--J`, `--sigma` arguments (previously hardcoded)
- Added `--disable_wb` to disable wandb logging

### 9. `main_ST.py` — Minor Updates

Updated to be consistent with graph_learning API changes.

### 10. `pyproject.toml` — New Dependencies

- `scipy>=1.14.0` — sparse matrices, pdist, eigsh (used in experiments and simplicial complex)
- `pytest>=8.0.0` — test runner (dev dependency)

Install with: `uv sync --dev`

---

## Experiments

All experiments are standalone scripts that save figures to `experiments/figures/`. No training or GPU required — pure math/spectral analysis.

### Shared Utilities

- `experiments/utils/manifold_samplers.py` — `sample_torus(N, R, r)`, `sample_sphere(N, radius)`, `BETTI_NUMBERS`
- `experiments/utils/laplacian_utils.py` — `build_rips_laplacians()`, `harmonic_subspace()`, `get_betti_1()`, `first_spectral_gap()`, `compute_diffusion_sq_dists()`

### Exp 1 — Geometry Sensitivity (`experiments/exp1_spectral_analysis.py`)

Compares geometric vs combinatorial Hodge 1-Laplacian on tori with different aspect ratios. Shows that the geometric Laplacian's eigenvalue spectrum reflects actual shape differences.

```bash
uv run python experiments/exp1_spectral_analysis.py
```
**Figures:** `exp1_eigenvalue_ratios.png`, `exp1_geometry_sensitivity.png`

### Exp 2 — Spectral Convergence (`experiments/exp2_spectral_convergence.py`)

Measures how fast eigenvalue ratios converge as sample size N increases. Geometric Laplacian should converge faster.

```bash
uv run python experiments/exp2_spectral_convergence.py
```
**Figures:** `exp2_convergence.png`, `exp2_ratio_profiles.png`

### Exp 3 — Edge Flow Prediction (`experiments/exp3_edge_flow_prediction.py`)

Hodge-regularized interpolation of edge flows on a torus. Demonstrates that the geometric Laplacian improves signal recovery from partial observations.

```bash
uv run python experiments/exp3_edge_flow_prediction.py
```
**Figures:** `exp3_mse_vs_lambda.png`, `exp3_mse_vs_train_fraction.png`

### Exp 4 — HiPoNet Training Comparison (`experiments/exp4_hiponet_comparison.py`)

Binary classification (torus vs sphere) using HiPoNet K=2. Trains two models (combinatorial vs geometric) and compares test accuracy.

```bash
uv run python experiments/exp4_hiponet_comparison.py
```
**Figures:** `exp4_accuracy_vs_epoch.png`, `exp4_loss_vs_epoch.png`, `exp4_final_accuracy.png`

### Exp 5 — Threshold Selection (`experiments/exp5_threshold_selection.py`)

Visualizes the adaptive threshold selection (Kneedle algorithm) — pairwise distance histograms and the edge-count knee curve.

```bash
uv run python experiments/exp5_threshold_selection.py
```
**Figures:** `exp5_distance_histograms.png`, `exp5_knee_curves.png`

---

## Tests

55 tests total, organized across 4 test files:

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Individual test files
uv run python -m pytest tests/test_geometric_laplacian.py -v   # 31 tests — numpy Laplacian, SimplicialComplex
uv run python -m pytest tests/test_diffusion_laplacian.py -v   # 24 tests — torch functions (differentiability, correctness)
uv run python -m pytest tests/test_attention.py -v             # PoolingAttention, attention in SimplicialFeatLearningLayer
uv run python -m pytest tests/test_threshold_selection.py -v   # Kneedle threshold selection
```

---

## SLURM Job Script

`regression_job.sh` — submits a regression job to the GPU cluster:

```bash
sbatch regression_job.sh --raw_dir melanoma_data_full --K 2 --use_geometric_laplacian
```

Requests: 1x H200 GPU, 200G RAM, 8 CPUs, 24h time limit.

---

## Gradient Flow (Geometric Path)

The full differentiable path for geometric Laplacian training:

```
alpha → X_weighted → Gaussian kernel → threshold → row-normalize → P
→ P^t (matrix_power) → squared diffusion distances
→ cayley_menger_volumes → geometric_hodge_laplacian_torch
→ transition matrices (P_U, P_L) → wavelet coefficients → MLP → loss
```

No `detach()`, no numpy, no eigendecomposition in the gradient path.
