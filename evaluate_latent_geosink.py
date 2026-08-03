#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr


def pair_metrics(latent: np.ndarray, target: np.ndarray) -> dict[str, float]:
    latent_vec = pdist(latent)
    target_vec = target[np.triu_indices(len(target), 1)]
    latent_norm = latent_vec / max(float(latent_vec.mean()), 1e-8)
    target_norm = target_vec / max(float(target_vec.mean()), 1e-8)
    return {
        "stress": float(np.sqrt(np.mean((latent_norm - target_norm) ** 2))),
        "pearson": float(np.corrcoef(latent_vec, target_vec)[0, 1]),
        "spearman": float(spearmanr(latent_vec, target_vec).statistic),
    }


def knn_overlap(latent: np.ndarray, target: np.ndarray, k: int = 5) -> float:
    k = min(k, len(latent) - 1)
    latent_d = cdist(latent, latent)
    np.fill_diagonal(latent_d, np.inf)
    target_d = target.copy()
    np.fill_diagonal(target_d, np.inf)
    latent_nn = np.argsort(latent_d, axis=1)[:, :k]
    target_nn = np.argsort(target_d, axis=1)[:, :k]
    return float(np.mean([len(set(a).intersection(b)) / k for a, b in zip(latent_nn, target_nn)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--training_summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    latents = np.load(args.latents).astype(np.float64)
    distances = np.load(args.distances).astype(np.float64)
    training = json.loads(args.training_summary.read_text())
    train_idx = np.asarray(training["train_indices"], dtype=np.int64)
    val_idx = np.asarray(training["val_indices"], dtype=np.int64)

    metrics = {
        "all": pair_metrics(latents, distances),
        "train": pair_metrics(latents[train_idx], distances[np.ix_(train_idx, train_idx)]),
        "validation": pair_metrics(latents[val_idx], distances[np.ix_(val_idx, val_idx)]),
        "all_knn5_overlap": knn_overlap(latents, distances, 5),
        "train_knn5_overlap": knn_overlap(
            latents[train_idx], distances[np.ix_(train_idx, train_idx)], 5
        ),
        "validation_knn3_overlap": knn_overlap(
            latents[val_idx], distances[np.ix_(val_idx, val_idx)], 3
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
