"""Exp 4: HiPoNet K=2 classification — combinatorial vs geometric Hodge Laplacian.

Binary classification: torus (label=0) vs sphere (label=1) point clouds.

The torus has β₁=2 and the sphere has β₁=0. A model that uses the geometric
Hodge Laplacian (metric-aware mass matrices from diffusion distances) should
produce richer edge-flow features that better separate the two manifolds.

Figures saved:
  exp4_accuracy_vs_epoch.png  — test accuracy per epoch, averaged over seeds
  exp4_loss_vs_epoch.png      — train loss per epoch, averaged over seeds
  exp4_final_accuracy.png     — bar chart of best test accuracy (mean ± std)

Run from project root:
    uv run python experiments/exp4_hiponet_comparison.py
"""

import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.graph_learning import HiPoNet, MLP
from utils.training import collate_fn

sys.path.insert(0, os.path.join(_here, "utils"))
from manifold_samplers import sample_torus, sample_sphere

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_PTS        = 70       # points per cloud (kept small — Rips is O(N³))
N_PTS_JITTER = 10       # vary per sample by ±JITTER to exercise padding
N_SAMPLES    = 60       # samples per class (60 torus + 60 sphere = 120 total)
N_EPOCHS     = 30
BATCH_SIZE   = 6
LR           = 5e-3
WD           = 1e-3
N_WEIGHTS    = 1        # learnable alpha vectors
J            = 3        # wavelet scales
K            = 2        # simplicial dimension (edges + triangles)
SIGMA        = 1.0      # Gaussian kernel bandwidth (for squared dists)
THRESHOLD    = 0.2      # adjacency threshold
HIDDEN_DIM   = 64
N_LAYERS     = 2
DIFFUSION_STEPS = 1
NOISE_STD    = 0.02     # point-cloud noise
SEEDS        = [0, 1, 2, 3, 4]

FIGURES_DIR = os.path.join(_here, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_dataset(seed_offset=0):
    """Generate N_SAMPLES torus + N_SAMPLES sphere point clouds.

    Point cloud sizes vary by ±N_PTS_JITTER to exercise padding in collate_fn.
    """
    rng = np.random.default_rng(seed_offset)
    data = []
    for i in range(N_SAMPLES):
        n = N_PTS + int(rng.integers(-N_PTS_JITTER, N_PTS_JITTER + 1))
        pts = sample_torus(n, R=3.0, r=1.0, noise_std=NOISE_STD, seed=seed_offset + i)
        data.append((torch.tensor(pts, dtype=torch.float32), 0))
    for i in range(N_SAMPLES):
        n = N_PTS + int(rng.integers(-N_PTS_JITTER, N_PTS_JITTER + 1))
        pts = sample_sphere(n, radius=2.0, noise_std=NOISE_STD, seed=seed_offset + N_SAMPLES + i)
        data.append((torch.tensor(pts, dtype=torch.float32), 1))
    return data


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def evaluate(model, mlp, loader):
    model.eval()
    mlp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, mask, labels in loader:
            batch, mask = batch.to(DEVICE), mask.to(DEVICE)
            X = model(batch, mask)
            logits = mlp(X)
            labels = labels.to(logits.device)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return 100.0 * correct / total if total > 0 else 0.0


def run_one_seed(seed, use_geometric):
    """Train HiPoNet K=2 on torus/sphere classification for one random seed.

    Returns
    -------
    test_accs : list of float, length N_EPOCHS
    train_losses : list of float, length N_EPOCHS
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = make_dataset(seed_offset=seed * 200)

    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=seed, stratify=[d[1] for d in data]
    )

    train_loader = DataLoader(
        [data[i] for i in train_idx],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, transpose=False),
    )
    test_loader = DataLoader(
        [data[i] for i in test_idx],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, transpose=False),
    )

    hiponet = HiPoNet(
        dimension=3,          # point clouds are 3D
        n_weights=N_WEIGHTS,
        threshold=THRESHOLD,
        K=K,
        J=J,
        device=DEVICE,
        sigma=SIGMA,
        pooling=True,
        use_geometric_laplacian=use_geometric,
        diffusion_steps=DIFFUSION_STEPS,
    ).to(DEVICE)

    # Infer MLP input dimension from a forward pass
    with torch.no_grad():
        sample_pc, _ = data[0]
        pc_batch = sample_pc.unsqueeze(0).to(DEVICE)   # (1, N, 3)
        n_pts = sample_pc.shape[0]
        mask_sample = torch.ones((1, n_pts), dtype=torch.bool, device=DEVICE)
        feat_dim = hiponet(pc_batch, mask_sample).shape[1]

    mlp = MLP(feat_dim, HIDDEN_DIM, 2, N_LAYERS).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(hiponet.parameters()) + list(mlp.parameters()),
        lr=LR, weight_decay=WD,
    )
    loss_fn = nn.CrossEntropyLoss()

    test_accs = []
    train_losses = []
    for epoch in range(N_EPOCHS):
        hiponet.train()
        mlp.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch, mask, labels in train_loader:
            batch, mask = batch.to(DEVICE), mask.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            X = hiponet(batch, mask)
            logits = mlp(X)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += len(labels)

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = 100.0 * epoch_correct / max(epoch_total, 1)
        acc = evaluate(hiponet, mlp, test_loader)
        test_accs.append(acc)
        train_losses.append(train_loss)

        print(f"  epoch {epoch+1:3d}/{N_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"test_acc={acc:.1f}%", flush=True)

    return test_accs, train_losses


# ---------------------------------------------------------------------------
# Main: run both methods over all seeds
# ---------------------------------------------------------------------------

results = {"comb": [], "geom": []}
losses = {"comb": [], "geom": []}

for seed in SEEDS:
    for key, use_geom in [("comb", False), ("geom", True)]:
        label = "geometric" if use_geom else "combinatorial"
        print(f"\n--- seed={seed}  {label} Laplacian ---", flush=True)
        t0 = time.time()
        accs, tloss = run_one_seed(seed, use_geom)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s  |  best test acc = {max(accs):.1f}%", flush=True)
        results[key].append(accs)
        losses[key].append(tloss)

# ---------------------------------------------------------------------------
# Plot 1: accuracy vs epoch
# ---------------------------------------------------------------------------

arr_c = np.array(results["comb"])   # (n_seeds, n_epochs)
arr_g = np.array(results["geom"])

mean_c = arr_c.mean(0)
std_c  = arr_c.std(0)
mean_g = arr_g.mean(0)
std_g  = arr_g.std(0)

epochs = np.arange(1, N_EPOCHS + 1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs, mean_c, "o-", color="steelblue",  lw=2, ms=4, label="Combinatorial Δ₁")
ax.fill_between(epochs, mean_c - std_c, mean_c + std_c, alpha=0.25, color="steelblue")
ax.plot(epochs, mean_g, "s-", color="darkorange", lw=2, ms=4, label="Geometric Δ̃₁")
ax.fill_between(epochs, mean_g - std_g, mean_g + std_g, alpha=0.25, color="darkorange")
ax.axhline(50, color="gray", ls="--", lw=1, label="Random baseline (50%)")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Test accuracy (%)", fontsize=12)
ax.set_title(
    f"HiPoNet K=2 — torus vs sphere classification\n"
    f"(N≈{N_PTS} pts/cloud, {N_SAMPLES} per class, {len(SEEDS)} seeds)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.set_ylim(0, 105)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "exp4_accuracy_vs_epoch.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"\nSaved {path}")

# ---------------------------------------------------------------------------
# Plot 2: train loss vs epoch
# ---------------------------------------------------------------------------

loss_c = np.array(losses["comb"])   # (n_seeds, n_epochs)
loss_g = np.array(losses["geom"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs, loss_c.mean(0), "o-", color="steelblue",  lw=2, ms=4, label="Combinatorial Δ₁")
ax.fill_between(epochs, loss_c.mean(0) - loss_c.std(0), loss_c.mean(0) + loss_c.std(0),
                alpha=0.25, color="steelblue")
ax.plot(epochs, loss_g.mean(0), "s-", color="darkorange", lw=2, ms=4, label="Geometric Δ̃₁")
ax.fill_between(epochs, loss_g.mean(0) - loss_g.std(0), loss_g.mean(0) + loss_g.std(0),
                alpha=0.25, color="darkorange")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Train loss", fontsize=12)
ax.set_title(
    f"HiPoNet K=2 — train loss\n"
    f"(N≈{N_PTS} pts/cloud, {N_SAMPLES} per class, {len(SEEDS)} seeds)",
    fontsize=11,
)
ax.legend(fontsize=10)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "exp4_loss_vs_epoch.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved {path}")

# ---------------------------------------------------------------------------
# Plot 3: best accuracy bar chart
# ---------------------------------------------------------------------------

best_c = arr_c.max(axis=1)
best_g = arr_g.max(axis=1)

fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(
    ["Combinatorial Δ₁", "Geometric Δ̃₁"],
    [best_c.mean(), best_g.mean()],
    yerr=[best_c.std(), best_g.std()],
    color=["steelblue", "darkorange"],
    edgecolor="k", linewidth=0.8, capsize=10, width=0.5,
)
for i, (mean, std) in enumerate([(best_c.mean(), best_c.std()), (best_g.mean(), best_g.std())]):
    ax.text(i, mean + std + 1.5, f"{mean:.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.axhline(50, color="gray", ls="--", lw=1)
ax.set_ylabel("Best test accuracy (%)", fontsize=12)
ax.set_title(
    f"HiPoNet K=2 — best accuracy\n(torus vs sphere, {len(SEEDS)} seeds)",
    fontsize=11,
)
ax.set_ylim(0, 115)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "exp4_final_accuracy.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved {path}")

print(f"\nSummary:")
print(f"  Combinatorial: best acc = {best_c.mean():.1f} ± {best_c.std():.1f}%")
print(f"  Geometric:     best acc = {best_g.mean():.1f} ± {best_g.std():.1f}%")
print("\nExp 4 complete.")
