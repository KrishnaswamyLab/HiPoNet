"""Experiment 5 — Adaptive Threshold Selection via Kneedle Algorithm.

Shows the distribution of pairwise distances in a point cloud and where
the auto-selected epsilon/threshold cuts off — highlighting that it lands
at a natural gap in the distance distribution.

Figures:
  1. exp5_distance_histograms.png — pairwise distance histograms for
     torus/sphere with epsilon cutoff as a vertical line, and the
     resulting included/excluded edges visualized on the point cloud
  2. exp5_knee_curve.png — the edge-count sweep curve + knee point
     (diagnostic / secondary)

Run:
    uv run python experiments/exp5_threshold_selection.py
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from experiments.utils.manifold_samplers import sample_torus, sample_sphere
from models.threshold_selection import select_threshold

FIGURE_DIR = os.path.join(_root, "experiments", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

SIGMA = 1.5
N = 120
SEED = 42


def _pairwise_dists(pts):
    """Compute upper-triangle pairwise distances (N*(N-1)/2,)."""
    pc = torch.tensor(pts, dtype=torch.float32)
    G = pc @ pc.T
    diag = G.diag().unsqueeze(0)
    D_sq = (diag + diag.T - 2 * G).clamp_min(0)
    D = D_sq.sqrt()
    n = pc.shape[0]
    i, j = torch.triu_indices(n, n, offset=1)
    return D[i, j].numpy(), pc, D_sq


def _epsilon_from_threshold(threshold, sigma):
    """Convert kernel threshold → distance cutoff.

    Edge exists when exp(-d²/sigma) >= threshold
    ⟺  d  <=  sqrt(-sigma * ln(threshold))
    """
    return np.sqrt(-sigma * np.log(threshold))


def plot_distance_histograms():
    """Main figure: pairwise distance histograms + epsilon line + point cloud."""

    manifolds = [
        ("Torus", sample_torus(N, seed=SEED)),
        ("Sphere", sample_sphere(N, seed=SEED)),
    ]

    fig = plt.figure(figsize=(16, 10))
    # 2 rows × 3 cols: [3D point cloud | distance histogram | edge count curve]
    outer = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for row, (name, pts) in enumerate(manifolds):
        dists, pc, D_sq = _pairwise_dists(pts)
        pc_np = pts

        # --- Run adaptive threshold selection ---
        th, data = select_threshold(pc, sigma=SIGMA, n_steps=80)
        eps = _epsilon_from_threshold(th, SIGMA)
        knee_idx = data["knee_idx"]
        thresholds = data["thresholds"]
        n_edges = data["n_edges"]

        # ── Col 0: 3D point cloud with included edges ──────────────────────
        ax3d = fig.add_subplot(outer[row, 0], projection="3d")
        ax3d.scatter(
            pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
            s=15, c="steelblue", alpha=0.7, depthshade=True,
        )

        # Draw included edges
        n = pc_np.shape[0]
        W = torch.exp(-D_sq / SIGMA)
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        edge_mask = W[i_idx, j_idx] >= th
        ei = i_idx[edge_mask].numpy()
        ej = j_idx[edge_mask].numpy()
        # Subsample for clarity
        rng = np.random.default_rng(0)
        sample = rng.choice(len(ei), size=min(300, len(ei)), replace=False)
        for s in sample:
            a, b = ei[s], ej[s]
            ax3d.plot(
                [pc_np[a, 0], pc_np[b, 0]],
                [pc_np[a, 1], pc_np[b, 1]],
                [pc_np[a, 2], pc_np[b, 2]],
                "b-", alpha=0.12, linewidth=0.5,
            )

        ax3d.set_title(
            f"{name}\n{int(edge_mask.sum())} edges included",
            fontsize=11, fontweight="bold",
        )
        ax3d.set_xticklabels([])
        ax3d.set_yticklabels([])
        ax3d.set_zticklabels([])

        # ── Col 1: Distance histogram + epsilon cutoff ─────────────────────
        ax_hist = fig.add_subplot(outer[row, 1])

        included = dists[dists <= eps]
        excluded = dists[dists > eps]

        ax_hist.hist(
            included, bins=40, color="steelblue", alpha=0.75,
            label=f"Included ({len(included):,})", density=True,
        )
        ax_hist.hist(
            excluded, bins=40, color="lightgray", alpha=0.75,
            label=f"Excluded ({len(excluded):,})", density=True,
        )
        ax_hist.axvline(
            eps, color="crimson", linewidth=2.5, linestyle="--",
            label=f"ε = {eps:.2f}  (θ={th:.3f})",
        )

        ax_hist.set_xlabel("Pairwise Distance", fontsize=11)
        ax_hist.set_ylabel("Density", fontsize=11)
        ax_hist.set_title(f"{name} — Distance Distribution", fontsize=12)
        ax_hist.legend(fontsize=9, loc="upper right")
        ax_hist.grid(True, alpha=0.3)

        # ── Col 2: Edge count sweep + knee ─────────────────────────────────
        ax_sweep = fig.add_subplot(outer[row, 2])
        eps_vals = np.array([_epsilon_from_threshold(t, SIGMA) for t in thresholds])

        ax_sweep.plot(eps_vals, n_edges, color="steelblue", linewidth=2)
        ax_sweep.axvline(
            eps, color="crimson", linewidth=2.5, linestyle="--",
            label=f"Knee ε={eps:.2f}",
        )
        ax_sweep.scatter(
            [eps], [n_edges[knee_idx]], color="crimson", s=80, zorder=5,
        )
        ax_sweep.set_xlabel("Distance Cutoff ε", fontsize=11)
        ax_sweep.set_ylabel("Number of Edges", fontsize=11)
        ax_sweep.set_title(f"{name} — Edge Count vs ε", fontsize=12)
        ax_sweep.legend(fontsize=9)
        ax_sweep.grid(True, alpha=0.3)

    fig.suptitle(
        f"Adaptive ε Selection via Kneedle  (N={N}, σ={SIGMA})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    path = os.path.join(FIGURE_DIR, "exp5_distance_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("=== Experiment 5: Adaptive Threshold Selection ===\n")
    print("[1/1] Distance histograms with epsilon cutoff...")
    plot_distance_histograms()
    print("\nDone! Figures saved to experiments/figures/")
