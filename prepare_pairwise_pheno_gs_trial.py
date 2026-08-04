#!/usr/bin/env python
"""Compute a memory-bounded Pheno-GS trial using HiPoNet-style Gaussian graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.special import ive
from sklearn.neighbors import NearestNeighbors


def heat_action(laplacian, values, phi, coefficients, order):
    t0 = values
    result = 0.5 * coefficients[0] * t0
    t1 = (laplacian @ values) / phi - t0
    result = result + coefficients[1] * t1
    for degree in range(2, order + 1):
        t2 = 2.0 * (laplacian @ t1) / phi - 2.0 * t1 - t0
        result = result + coefficients[degree] * t2
        t0, t1 = t1, t2
    return np.asarray(result).reshape(-1)


def gaussian_laplacian(points, knn, sigma, threshold):
    n_neighbors = min(knn + 1, len(points))
    model = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1).fit(points)
    distances, indices = model.kneighbors(points)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    rows = np.repeat(np.arange(len(points)), indices.shape[1])
    cols = indices.reshape(-1)
    weights = np.exp(-(distances.reshape(-1) ** 2) / sigma)
    keep = weights >= threshold
    from scipy.sparse import csr_matrix

    adjacency = csr_matrix(
        (weights[keep], (rows[keep], cols[keep])),
        shape=(len(points), len(points)),
    )
    adjacency = adjacency.maximum(adjacency.T)
    if adjacency.nnz == 0:
        raise ValueError("Gaussian threshold removed every graph edge")
    return csgraph.laplacian(adjacency, normed=False).tocsr()


def directed_distance(
    laplacian,
    source_rows,
    target_rows,
    phi,
    coefficients,
    order,
    diffusion_time,
    epsilon,
):
    n_cells = laplacian.shape[0]
    base_mass = np.full(n_cells, 1.0 / n_cells, dtype=np.float64)
    base_diffusion = np.maximum(
        heat_action(laplacian, base_mass, phi, coefficients, order),
        epsilon,
    )
    source_mass = np.full(len(source_rows), 1.0 / len(source_rows))
    source_ratio = source_mass / base_diffusion[source_rows]
    sum_source = float(
        (np.log(np.clip(source_ratio, epsilon, 1e32) + epsilon) * source_mass).sum()
    )
    source_measure = np.zeros(n_cells, dtype=np.float64)
    source_measure[source_rows] = source_ratio
    source_measure *= base_mass[0]
    diffused = np.maximum(
        heat_action(laplacian, source_measure, phi, coefficients, order),
        epsilon,
    )
    target_mass = np.full(len(target_rows), 1.0 / len(target_rows))
    target_ratio = target_mass / diffused[target_rows]
    cross = float(
        (np.log(np.clip(target_ratio, epsilon, 1e32) + epsilon) * target_mass).sum()
    )
    return max(4.0 * base_mass[0] * diffusion_time * (sum_source + cross), 0.0)


def population_pair_distance(
    index_i,
    index_j,
    populations,
    knn,
    sigma,
    threshold,
    diffusion_time,
    order,
    epsilon,
):
    population_i = populations[index_i]
    population_j = populations[index_j]
    points = np.concatenate((population_i, population_j), axis=0)
    laplacian = gaussian_laplacian(points, knn, sigma, threshold)
    largest = float(
        eigsh(laplacian, k=1, which="LM", return_eigenvectors=False)[0]
    )
    phi = largest / 2.0
    coefficients = 2.0 * ive(
        np.arange(order + 1), -diffusion_time * phi
    )
    rows_i = np.arange(len(population_i), dtype=np.int32)
    rows_j = np.arange(len(population_i), len(points), dtype=np.int32)
    forward = directed_distance(
        laplacian,
        rows_i,
        rows_j,
        phi,
        coefficients,
        order,
        diffusion_time,
        epsilon,
    )
    reverse = directed_distance(
        laplacian,
        rows_j,
        rows_i,
        phi,
        coefficients,
        order,
        diffusion_time,
        epsilon,
    )
    return index_i, index_j, 0.5 * (forward + reverse)


def subset_cache(cache, selected, output_path):
    payload = {}
    n_populations = len(cache["populations"])
    for key in cache.files:
        value = np.asarray(cache[key])
        if value.ndim > 0 and len(value) == n_populations:
            payload[key] = value[selected]
        else:
            payload[key] = value
    payload["source_population_indices"] = selected
    np.savez_compressed(output_path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--n_populations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1302)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=32.0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--diffusion_time", type=float, default=10.0)
    parser.add_argument("--chebyshev_order", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.cache, allow_pickle=True)
    total_populations = len(cache["populations"])
    if not 3 <= args.n_populations <= total_populations:
        raise ValueError("--n_populations must be between 3 and the cache size")
    selected = np.sort(
        np.random.default_rng(args.seed).choice(
            total_populations, args.n_populations, replace=False
        )
    )
    populations = [
        np.asarray(cache["populations"][index], dtype=np.float64)
        for index in selected
    ]
    names = [str(cache["group_names"][index]) for index in selected]
    subset_path = args.out_dir / "population_subset.npz"
    subset_cache(cache, selected, subset_path)

    pairs = [
        (i, j)
        for i in range(args.n_populations)
        for j in range(i)
    ]
    print(
        f"Computing {len(pairs)} pairwise Pheno-GS targets from "
        f"{args.n_populations} populations using HiPoNet-style Gaussian graphs",
        flush=True,
    )
    results = Parallel(n_jobs=args.n_jobs, prefer="threads", verbose=10)(
        delayed(population_pair_distance)(
            i,
            j,
            populations,
            args.knn,
            args.sigma,
            args.threshold,
            args.diffusion_time,
            args.chebyshev_order,
            args.epsilon,
        )
        for i, j in pairs
    )
    distances = np.zeros((args.n_populations, args.n_populations), dtype=np.float64)
    for i, j, distance in results:
        distances[i, j] = distances[j, i] = distance
    if not np.isfinite(distances).all():
        raise ValueError("Pairwise Pheno-GS produced non-finite distances")

    distance_path = args.out_dir / "pheno_gs_distances.npy"
    np.save(distance_path, distances.astype(np.float32))
    pd.DataFrame(distances, index=names, columns=names).to_csv(
        args.out_dir / "pheno_gs_distances.csv"
    )
    upper = distances[np.triu_indices(args.n_populations, 1)]
    summary = {
        "method": "pairwise Pheno-GS with HiPoNet-style Gaussian graphs",
        "cache": str(args.cache),
        "subset_cache": str(subset_path),
        "source_population_indices": selected.tolist(),
        "n_populations": args.n_populations,
        "n_cells": int(sum(len(population) for population in populations)),
        "n_features": int(populations[0].shape[1]),
        "knn": args.knn,
        "sigma": args.sigma,
        "threshold": args.threshold,
        "diffusion_time": args.diffusion_time,
        "chebyshev_order": args.chebyshev_order,
        "distance_min": float(upper.min()),
        "distance_median": float(np.median(upper)),
        "distance_mean": float(upper.mean()),
        "distance_max": float(upper.max()),
        "distance_output": str(distance_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
