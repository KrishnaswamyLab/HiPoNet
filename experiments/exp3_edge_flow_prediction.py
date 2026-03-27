"""Exp 3: Edge flow prediction via Hodge-regularized interpolation on a torus.

Ground-truth edge flow f* = B₁ᵀ φ + h  (gradient + harmonic component).
A fraction of edges are observed; the rest are predicted via:
    (A + λΔ₁ + εI) f̂ = A f_obs
where A = diag(train_mask).

Figures saved:
  exp3_mse_vs_lambda.png        — MSE vs λ at a fixed train fraction
  exp3_mse_vs_train_fraction.png — MSE vs train fraction (optimal λ)

Run from project root:
    uv run python experiments/exp3_edge_flow_prediction.py
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, os.path.join(_here, "utils"))

import numpy as np
import time
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve

from manifold_samplers import sample_torus
from laplacian_utils import build_rips_laplacians

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N = 300
DIFFUSION_STEPS = 1
NOISE_STD = 0.02          # point-cloud noise
SIGNAL_NOISE = 0.05       # noise added to ground-truth flow
EPSILON = 1e-4            # ridge for numerical stability
LAMBDAS = np.logspace(-3, 2, 40)
TRAIN_FRACS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIXED_TRAIN_FRAC = 0.7
N_TRIALS = 5

FIGURES_DIR = os.path.join(_here, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Prediction via Hodge-regularized system
# ---------------------------------------------------------------------------
def hodge_predict(L, f_obs, train_mask, lam, eps=EPSILON):
    """Solve (A + λΔ₁ + εI) f = A f_obs where A = diag(train_mask)."""
    A = np.diag(train_mask.astype(float))
    M = A + lam * L + eps * np.eye(L.shape[0])
    rhs = A @ f_obs
    return solve(M, rhs, assume_a="gen")


def best_mse_lambda(L, f_true, f_obs, train_mask, lambdas):
    test_mask = ~train_mask
    mses = []
    for lam in lambdas:
        f_hat = hodge_predict(L, f_obs, train_mask, lam)
        err = f_hat[test_mask] - f_true[test_mask]
        mses.append(np.mean(err ** 2))
    mses = np.array(mses)
    best = mses.argmin()
    return mses[best], lambdas[best], mses


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print(f"Edge flow prediction on torus  (N={N}, {N_TRIALS} trials)", flush=True)
print(f"Config: SIGNAL_NOISE={SIGNAL_NOISE}, EPSILON={EPSILON}, "
      f"FIXED_TRAIN_FRAC={FIXED_TRAIN_FRAC}, "
      f"n_lambdas={len(LAMBDAS)}, n_fracs={len(TRAIN_FRACS)}", flush=True)

results_frac = {"comb": [], "geom": []}   # (n_trials, n_fracs)
lambda_curves_fixed = {"comb": None, "geom": None}

for trial in range(N_TRIALS):
    t_trial = time.time()
    print(f"\n--- trial {trial}/{N_TRIALS-1} ---", flush=True)

    try:
        pts = sample_torus(N, seed=trial, R=3.0, r=1.0, noise_std=NOISE_STD)
        print(f"  sampled {N} points  [mem ok]", flush=True)

        L_comb, L_geom, sc, vols, B1d, _ = build_rips_laplacians(
            pts, diffusion_steps=DIFFUSION_STEPS
        )
    except Exception:
        print(f"  ERROR during complex build:", flush=True)
        traceback.print_exc()
        continue

    n_edges = L_comb.shape[0]
    if n_edges == 0:
        print(f"  SKIP: empty complex", flush=True)
        continue

    print(f"  complex: {sc.num_simplices(1)} edges, {sc.num_simplices(2)} triangles  "
          f"L_comb shape={L_comb.shape}", flush=True)

    # Ground-truth: pure gradient signal f* = B₁ᵀ φ
    # (harmonic component excluded — it's in null(Δ₁) and cannot be predicted
    #  by Hodge regularization from partial observations)
    rng = np.random.default_rng(trial + 100)
    phi = rng.standard_normal(B1d.shape[0])
    f_true = B1d.T @ phi
    f_true /= (np.linalg.norm(f_true) + 1e-12)
    f_obs = f_true + rng.standard_normal(n_edges) * SIGNAL_NOISE
    print(f"  f_true: pure gradient, norm=1.000  "
          f"||f_obs - f_true||={np.linalg.norm(f_obs - f_true):.4f}", flush=True)

    # --- MSE vs train fraction ---
    print(f"  sweeping {len(TRAIN_FRACS)} train fractions × {len(LAMBDAS)} lambdas ...",
          flush=True)
    t_sweep = time.time()
    row_c = []
    row_g = []
    for frac in TRAIN_FRACS:
        idx = rng.choice(n_edges, size=int(frac * n_edges), replace=False)
        mask = np.zeros(n_edges, dtype=bool)
        mask[idx] = True
        m_c, lam_c, _ = best_mse_lambda(L_comb, f_true, f_obs, mask, LAMBDAS)
        m_g, lam_g, _ = best_mse_lambda(L_geom, f_true, f_obs, mask, LAMBDAS)
        row_c.append(m_c)
        row_g.append(m_g)
        print(f"    frac={frac:.1f}: comb mse={m_c:.4f} (λ*={lam_c:.2e})  "
              f"geom mse={m_g:.4f} (λ*={lam_g:.2e})", flush=True)
    results_frac["comb"].append(row_c)
    results_frac["geom"].append(row_g)
    print(f"  frac sweep done in {time.time()-t_sweep:.1f}s", flush=True)

    # --- MSE vs lambda at fixed train fraction ---
    print(f"  lambda curve at fixed frac={FIXED_TRAIN_FRAC} ...", flush=True)
    idx_f = rng.choice(n_edges, size=int(FIXED_TRAIN_FRAC * n_edges), replace=False)
    mask_f = np.zeros(n_edges, dtype=bool)
    mask_f[idx_f] = True
    _, _, curve_c = best_mse_lambda(L_comb, f_true, f_obs, mask_f, LAMBDAS)
    _, _, curve_g = best_mse_lambda(L_geom, f_true, f_obs, mask_f, LAMBDAS)
    lambda_curves_fixed["comb"] = curve_c
    lambda_curves_fixed["geom"] = curve_g
    print(f"  lambda curve done", flush=True)

    print(f"  trial {trial} complete in {time.time()-t_trial:.1f}s", flush=True)

# ---------------------------------------------------------------------------
# Plot 1: MSE vs train fraction
# ---------------------------------------------------------------------------
arr_c = np.array(results_frac["comb"])
arr_g = np.array(results_frac["geom"])
mean_c, std_c = arr_c.mean(0), arr_c.std(0)
mean_g, std_g = arr_g.mean(0), arr_g.std(0)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(TRAIN_FRACS, mean_c, "o-", color="steelblue",  lw=2, label="Combinatorial Δ₁")
ax.fill_between(TRAIN_FRACS, mean_c - std_c, mean_c + std_c, alpha=0.25, color="steelblue")
ax.plot(TRAIN_FRACS, mean_g, "s-", color="darkorange", lw=2, label="Geometric Δ̃₁")
ax.fill_between(TRAIN_FRACS, mean_g - std_g, mean_g + std_g, alpha=0.25, color="darkorange")
ax.set_xlabel("Train fraction", fontsize=12)
ax.set_ylabel("MSE on test edges (optimal λ)", fontsize=12)
ax.set_title(
    f"Edge flow prediction — MSE vs train fraction  (torus, N={N}, {N_TRIALS} trials)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.set_yscale("log")
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "exp3_mse_vs_train_fraction.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"\nSaved {path}")

# ---------------------------------------------------------------------------
# Plot 2: MSE vs lambda at fixed train fraction
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
if lambda_curves_fixed["comb"] is not None:
    ax.plot(LAMBDAS, lambda_curves_fixed["comb"], color="steelblue",  lw=2, label="Combinatorial Δ₁")
    ax.plot(LAMBDAS, lambda_curves_fixed["geom"], color="darkorange", lw=2, label="Geometric Δ̃₁")
ax.set_xlabel("Regularization λ", fontsize=12)
ax.set_ylabel("MSE on test edges", fontsize=12)
ax.set_title(
    f"Edge flow prediction — MSE vs λ  (train={FIXED_TRAIN_FRAC:.0%}, torus, N={N})",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.set_xscale("log")
ax.set_yscale("log")
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "exp3_mse_vs_lambda.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved {path}")

print("\nExp 3 complete.")
