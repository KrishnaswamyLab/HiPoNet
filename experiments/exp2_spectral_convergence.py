"""Exp 2: Spectral convergence — geometric vs combinatorial Hodge 1-Laplacian.

As the number of sample points N increases, the discrete Laplacian eigenvalues
should converge to stable values. The geometric Laplacian (metric-aware via
diffusion-distance volumes) should converge faster because it accounts for the
actual manifold geometry rather than just graph connectivity.

Experiment: fix a torus (R=3, r=1), sample at increasing N. Compute normalized
eigenvalue ratios λ_k / λ_1 and compare to a high-N reference. Measure relative
error vs N for both Laplacians.

Figures saved:
  exp2_convergence.png       — relative error vs N (comb vs geom)
  exp2_ratio_profiles.png    — ratio profiles at each N vs reference

Run from project root:
    uv run python experiments/exp2_spectral_convergence.py
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
R, r = 3.0, 1.0
N_SIZES = [50, 75, 100, 125, 150]
N_REF = 200
N_TRIALS = 5
K_MAX = 10
DIFFUSION_STEPS = 1
N_HARM = 2              # known β₁ for torus — bypass unreliable _gap_split

FIGURES_DIR = os.path.join(_here, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Config: R={R}, r={r}, N_SIZES={N_SIZES}, N_REF={N_REF}, "
      f"N_TRIALS={N_TRIALS}, K_MAX={K_MAX}", flush=True)

# ---------------------------------------------------------------------------
# Step 1: compute reference eigenvalue ratios at N_REF
# ---------------------------------------------------------------------------
print(f"\n=== Computing reference at N={N_REF} ===", flush=True)
ref_ratios = {"comb": [], "geom": []}

for trial in range(N_TRIALS):
    t0 = time.time()
    pts = sample_torus(N_REF, R=R, r=r, noise_std=0.0, seed=trial + 1000)
    L_comb, L_geom, sc, vols, B1d, B2d = build_rips_laplacians(
        pts, diffusion_steps=DIFFUSION_STEPS, max_triangles=None,
        verbose=False,
    )

    n_edges = L_comb.shape[0]
    n_tris = B2d.shape[1] if B2d.ndim == 2 else 0
    beta1_c = get_betti_1(L_comb)
    beta1_g = get_betti_1(L_geom)

    ratios_c = compute_eigenvalue_ratios(L_comb, K_MAX, n_harm=N_HARM)
    ratios_g = compute_eigenvalue_ratios(L_geom, K_MAX, n_harm=N_HARM)

    elapsed = time.time() - t0
    print(f"  trial {trial}: {n_edges} edges, {n_tris} tris, "
          f"β₁(c)={beta1_c} β₁(g)={beta1_g}  ({elapsed:.1f}s)", flush=True)

    if ratios_c is not None and ratios_g is not None:
        ref_ratios["comb"].append(ratios_c)
        ref_ratios["geom"].append(ratios_g)

if not ref_ratios["comb"]:
    print("ERROR: no valid reference trials. Exiting.", flush=True)
    sys.exit(1)

ref_comb = np.mean(ref_ratios["comb"], axis=0)  # (K_MAX,)
ref_geom = np.mean(ref_ratios["geom"], axis=0)

print(f"\nReference ratios (mean of {len(ref_ratios['comb'])} trials):", flush=True)
print(f"  comb: {np.array2string(ref_comb, precision=3)}", flush=True)
print(f"  geom: {np.array2string(ref_geom, precision=3)}", flush=True)

# ---------------------------------------------------------------------------
# Step 2: compute ratios at each N and measure relative error vs reference
# ---------------------------------------------------------------------------
# errors[method][N_idx] = list of relative errors (one per trial)
errors = {"comb": {i: [] for i in range(len(N_SIZES))},
          "geom": {i: [] for i in range(len(N_SIZES))}}
# Store ratio profiles for plotting
ratio_profiles = {"comb": {i: [] for i in range(len(N_SIZES))},
                  "geom": {i: [] for i in range(len(N_SIZES))}}

for ni, N in enumerate(N_SIZES):
    print(f"\n=== N={N} ===", flush=True)
    for trial in range(N_TRIALS):
        t0 = time.time()
        pts = sample_torus(N, R=R, r=r, noise_std=0.0, seed=trial)
        L_comb, L_geom, sc, vols, B1d, B2d = build_rips_laplacians(
            pts, diffusion_steps=DIFFUSION_STEPS, max_triangles=None,
            verbose=False,
        )

        ratios_c = compute_eigenvalue_ratios(L_comb, K_MAX, n_harm=N_HARM)
        ratios_g = compute_eigenvalue_ratios(L_geom, K_MAX, n_harm=N_HARM)

        elapsed = time.time() - t0

        if ratios_c is None or ratios_g is None:
            print(f"  trial {trial}: SKIP (not enough eigenvalues)  ({elapsed:.1f}s)",
                  flush=True)
            continue

        # Relative error: mean(|ratio - ref| / ref) over k
        err_c = float(np.mean(np.abs(ratios_c - ref_comb) / ref_comb))
        err_g = float(np.mean(np.abs(ratios_g - ref_geom) / ref_geom))
        errors["comb"][ni].append(err_c)
        errors["geom"][ni].append(err_g)
        ratio_profiles["comb"][ni].append(ratios_c)
        ratio_profiles["geom"][ni].append(ratios_g)

        print(f"  trial {trial}: err_comb={err_c:.4f}  err_geom={err_g:.4f}  "
              f"({elapsed:.1f}s)", flush=True)

# ---------------------------------------------------------------------------
# Plot 1: convergence curve — relative error vs N
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

for method, color, marker, label in [
    ("comb", "steelblue", "o", "Combinatorial Δ₁"),
    ("geom", "darkorange", "s", "Geometric Δ̃₁"),
]:
    means, stds, valid_N = [], [], []
    for ni, N in enumerate(N_SIZES):
        errs = errors[method][ni]
        if errs:
            means.append(np.mean(errs))
            stds.append(np.std(errs))
            valid_N.append(N)

    means = np.array(means)
    stds = np.array(stds)
    ax.plot(valid_N, means, f"{marker}-", color=color, lw=2, ms=6, label=label)
    ax.fill_between(valid_N, means - stds, np.maximum(means + stds, 1e-6),
                    alpha=0.2, color=color)

ax.set_xlabel("Sample size N", fontsize=12)
ax.set_ylabel("Relative error vs reference (N=1000)", fontsize=12)
ax.set_yscale("log")
ax.set_title(
    f"Spectral Convergence — Eigenvalue Ratio Error\n"
    f"(torus R={R}, r={r}, {N_TRIALS} trials, K={K_MAX} ratios)",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "exp2_convergence.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"\nSaved {path}", flush=True)

# ---------------------------------------------------------------------------
# Plot 2: ratio profiles at each N vs reference (2-panel)
# ---------------------------------------------------------------------------
k_indices = np.arange(2, K_MAX + 2)
cmap = plt.cm.viridis
n_colors = len(N_SIZES)

fig, (ax_c, ax_g) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ni, N in enumerate(N_SIZES):
    c = cmap(ni / max(n_colors - 1, 1))

    profiles_c = ratio_profiles["comb"][ni]
    profiles_g = ratio_profiles["geom"][ni]

    if profiles_c:
        mean_c = np.mean(profiles_c, axis=0)
        ax_c.plot(k_indices, mean_c, "o-", color=c, lw=1.5, ms=4, label=f"N={N}")
    if profiles_g:
        mean_g = np.mean(profiles_g, axis=0)
        ax_g.plot(k_indices, mean_g, "s-", color=c, lw=1.5, ms=4, label=f"N={N}")

# Reference lines
ax_c.plot(k_indices, ref_comb, "k--", lw=2, ms=0, label=f"Ref (N={N_REF})")
ax_g.plot(k_indices, ref_geom, "k--", lw=2, ms=0, label=f"Ref (N={N_REF})")

ax_c.set_title("Combinatorial Δ₁", fontsize=13)
ax_g.set_title("Geometric Δ̃₁", fontsize=13)

for ax in (ax_c, ax_g):
    ax.set_xlabel("Eigenvalue index k", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

ax_c.set_ylabel("Normalized ratio  λ_k / λ₁", fontsize=12)

fig.suptitle(
    f"Eigenvalue Ratio Profiles — Convergence with Increasing N\n"
    f"(torus R={R}, r={r}, dashed = reference at N={N_REF})",
    fontsize=12, y=1.02,
)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "exp2_ratio_profiles.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {path}", flush=True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nSummary:")
for method, label in [("comb", "Combinatorial"), ("geom", "Geometric")]:
    for ni, N in enumerate(N_SIZES):
        errs = errors[method][ni]
        if errs:
            print(f"  {label} N={N}: err={np.mean(errs):.4f} ± {np.std(errs):.4f}")
        else:
            print(f"  {label} N={N}: no valid trials")

print("\nExp 2 complete.", flush=True)
