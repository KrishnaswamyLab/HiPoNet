"""Exp 1: Geometry sensitivity — geometric vs combinatorial Hodge 1-Laplacian.

The geometric Laplacian incorporates metric information (simplex volumes from
diffusion distances), so its eigenvalue spectrum should reflect actual shape
differences. The combinatorial Laplacian depends only on graph connectivity,
which is similar across tori of different aspect ratios when the Rips threshold
adapts to point-cloud scale.

Experiment: generate tori with different R/r aspect ratios (same topology β₁=2,
different geometry). Compute normalized eigenvalue ratio profiles λ_k / λ_1 for
both Laplacians. The geometric Laplacian's ratios should vary more across
configurations — quantified by the coefficient of variation.

Figures saved:
  exp1_eigenvalue_ratios.png     — 2-panel ratio profiles (comb | geom)
  exp1_geometry_sensitivity.png  — CoV bar chart across configs

Run from project root:
    uv run python experiments/exp1_spectral_analysis.py
"""

import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, os.path.join(_here, "utils"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifold_samplers import sample_torus
from laplacian_utils import build_rips_laplacians, get_betti_1, compute_eigenvalue_ratios

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TORUS_CONFIGS = [
    {"R": 3.0, "r": 1.0, "label": "R=3, r=1"},
    {"R": 3.0, "r": 0.5, "label": "R=3, r=0.5"},
    {"R": 5.0, "r": 1.0, "label": "R=5, r=1"},
    {"R": 5.0, "r": 0.5, "label": "R=5, r=0.5"},
]
N = 150
N_TRIALS = 5
K_MAX = 10          # number of eigenvalue ratios to track
DIFFUSION_STEPS = 1

FIGURES_DIR = os.path.join(_here, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Config: N={N}, {len(TORUS_CONFIGS)} torus configs, "
      f"{N_TRIALS} trials, K_MAX={K_MAX}", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
# results[method][config_idx] = list of ratio arrays (one per successful trial)
results = {
    "comb": {i: [] for i in range(len(TORUS_CONFIGS))},
    "geom": {i: [] for i in range(len(TORUS_CONFIGS))},
}

for ci, cfg in enumerate(TORUS_CONFIGS):
    print(f"\n=== Config {ci}: {cfg['label']} ===", flush=True)
    for trial in range(N_TRIALS):
        t0 = time.time()
        pts = sample_torus(N, R=cfg["R"], r=cfg["r"], noise_std=0.0, seed=trial)

        try:
            L_comb, L_geom, sc, vols, B1d, B2d = build_rips_laplacians(
                pts, diffusion_steps=DIFFUSION_STEPS, max_triangles=None,
                verbose=False,
            )
        except Exception as e:
            print(f"  trial {trial}: build failed — {e}", flush=True)
            continue

        n_edges = L_comb.shape[0]
        n_tris = B2d.shape[1] if B2d.ndim == 2 else 0
        beta1_c = get_betti_1(L_comb)
        beta1_g = get_betti_1(L_geom)

        ratios_c = compute_eigenvalue_ratios(L_comb, K_MAX)
        ratios_g = compute_eigenvalue_ratios(L_geom, K_MAX)

        elapsed = time.time() - t0
        print(f"  trial {trial}: {n_edges} edges, {n_tris} tris, "
              f"β₁(comb)={beta1_c} β₁(geom)={beta1_g}  "
              f"({elapsed:.1f}s)", flush=True)

        if beta1_c != 2 or beta1_g != 2:
            print(f"    WARNING: β₁ ≠ 2 (may indicate incomplete complex)", flush=True)
        if ratios_c is None or ratios_g is None:
            print(f"    SKIP: not enough eigenvalues", flush=True)
            continue

        results["comb"][ci].append(ratios_c)
        results["geom"][ci].append(ratios_g)

# ---------------------------------------------------------------------------
# Compute means per config
# ---------------------------------------------------------------------------
mean_ratios = {"comb": {}, "geom": {}}
for method in ["comb", "geom"]:
    for ci in range(len(TORUS_CONFIGS)):
        arr = results[method][ci]
        if arr:
            mean_ratios[method][ci] = np.mean(arr, axis=0)
        else:
            mean_ratios[method][ci] = None

# Check we have enough data
valid_configs = [ci for ci in range(len(TORUS_CONFIGS))
                 if mean_ratios["comb"][ci] is not None
                 and mean_ratios["geom"][ci] is not None]
print(f"\nValid configs: {len(valid_configs)} / {len(TORUS_CONFIGS)}", flush=True)

if len(valid_configs) < 2:
    print("ERROR: Need at least 2 valid configs for comparison. Exiting.", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Plot 1: eigenvalue ratio profiles (2-panel)
# ---------------------------------------------------------------------------
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
k_indices = np.arange(2, K_MAX + 2)  # eigenvalue indices k=2..K_MAX+1

fig, (ax_c, ax_g) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ci in valid_configs:
    c = colors[ci % len(colors)]
    label = TORUS_CONFIGS[ci]["label"]

    r_c = mean_ratios["comb"][ci]
    r_g = mean_ratios["geom"][ci]

    ax_c.plot(k_indices, r_c, "o-", color=c, lw=2, ms=5, label=label)
    ax_g.plot(k_indices, r_g, "s-", color=c, lw=2, ms=5, label=label)

ax_c.set_title("Combinatorial Δ₁", fontsize=13)
ax_g.set_title("Geometric Δ̃₁", fontsize=13)

for ax in (ax_c, ax_g):
    ax.set_xlabel("Eigenvalue index k", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

ax_c.set_ylabel("Normalized ratio  λ_k / λ₁", fontsize=12)

fig.suptitle(
    f"Eigenvalue Ratio Profiles — Tori with Different Aspect Ratios\n"
    f"(N={N}, {N_TRIALS} trials, same topology β₁=2)",
    fontsize=12, y=1.02,
)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "exp1_eigenvalue_ratios.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {path}", flush=True)

# ---------------------------------------------------------------------------
# Plot 2: geometry sensitivity (coefficient of variation per eigenvalue index)
# ---------------------------------------------------------------------------
# For each eigenvalue index k, compute CoV of mean ratios across configs
cov_comb = np.zeros(K_MAX)
cov_geom = np.zeros(K_MAX)

stack_c = np.array([mean_ratios["comb"][ci] for ci in valid_configs])  # (n_configs, K_MAX)
stack_g = np.array([mean_ratios["geom"][ci] for ci in valid_configs])

for k in range(K_MAX):
    mu_c = stack_c[:, k].mean()
    mu_g = stack_g[:, k].mean()
    cov_comb[k] = stack_c[:, k].std() / mu_c if mu_c > 1e-12 else 0.0
    cov_geom[k] = stack_g[:, k].std() / mu_g if mu_g > 1e-12 else 0.0

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(K_MAX)
w = 0.35
ax.bar(x - w / 2, cov_comb, w, color="steelblue", edgecolor="k", lw=0.5,
       label="Combinatorial Δ₁")
ax.bar(x + w / 2, cov_geom, w, color="darkorange", edgecolor="k", lw=0.5,
       label="Geometric Δ̃₁")

ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in range(2, K_MAX + 2)], fontsize=9)
ax.set_xlabel("Eigenvalue index k", fontsize=12)
ax.set_ylabel("Coefficient of variation across configs", fontsize=12)
ax.set_title(
    f"Geometry Sensitivity — Eigenvalue Ratio Variability\n"
    f"(higher = more sensitive to shape changes)",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "exp1_geometry_sensitivity.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved {path}", flush=True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nSummary:")
print(f"  Mean CoV — combinatorial: {cov_comb.mean():.4f}")
print(f"  Mean CoV — geometric:     {cov_geom.mean():.4f}")
ratio = cov_geom.mean() / cov_comb.mean() if cov_comb.mean() > 1e-12 else float("inf")
print(f"  Geometric / Combinatorial sensitivity ratio: {ratio:.2f}x")
print("\nExp 1 complete.", flush=True)
