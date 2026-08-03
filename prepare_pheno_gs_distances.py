#!/usr/bin/env python
"""Prepare a population distance matrix using the supplied Pheno-GS method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import graphtools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigsh
from scipy.special import ive


def expm_multiply(laplacian, values, phi, coefficients, order):
    inv_phi = 1.0 / phi
    t0 = values
    result = 0.5 * coefficients[0] * t0
    t1 = inv_phi * (laplacian @ values) - t0
    result = result + coefficients[1] * t1
    for degree in range(2, order + 1):
        t2 = 2.0 * inv_phi * (laplacian @ t1) - 2.0 * t1 - t0
        result = result + coefficients[degree] * t2
        t0, t1 = t1, t2
    return np.asarray(result)


def chebyshev_coefficients(phi, diffusion_time, order):
    return 2.0 * ive(np.arange(order + 1), -diffusion_time * phi)


def process_condition(
    condition_j,
    laplacian,
    m_v_column,
    phi,
    coefficients,
    order,
    rows_per_condition,
    mass_per_condition,
    pair_indices,
    epsilon,
):
    diffused = expm_multiply(
        laplacian, m_v_column, phi, coefficients, order
    ).reshape(-1)
    results = []
    for condition_i in range(condition_j + 1, len(rows_per_condition)):
        rows_i = rows_per_condition[condition_i]
        mass_i = mass_per_condition[condition_i]
        denominator = np.maximum(diffused[rows_i], epsilon)
        ratio = np.clip(mass_i / denominator, epsilon, 1e32)
        contribution = float((np.log(ratio + epsilon) * mass_i).sum())
        results.append((pair_indices[condition_i, condition_j], contribution))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--diffusion_time", type=float, default=10.0)
    parser.add_argument("--chebyshev_order", type=int, default=5)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(args.cache, allow_pickle=True)
    populations = [np.asarray(value, dtype=np.float64) for value in cache["populations"]]
    names = [str(value) for value in cache["group_names"]]
    sizes = np.asarray([len(value) for value in populations], dtype=np.int64)
    if np.any(sizes == 0):
        raise ValueError("Pheno-GS requires every population to contain cells or landmarks")

    cells = np.concatenate(populations, axis=0)
    n_cells = len(cells)
    n_populations = len(populations)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    rows_per_condition = [
        np.arange(start, start + size, dtype=np.int32)
        for start, size in zip(starts, sizes)
    ]
    mass_per_condition = [
        np.full(size, 1.0 / size, dtype=np.float64) for size in sizes
    ]

    print(
        f"Building shared Pheno-GS cell graph: {n_cells} landmarks, "
        f"{n_populations} populations, {cells.shape[1]} features",
        flush=True,
    )
    graph = graphtools.Graph(
        cells, use_pygsp=True, knn=args.knn, n_jobs=args.n_jobs
    )
    graph.compute_laplacian("combinatorial")
    laplacian = graph.L.tocsr()

    largest_eigenvalue = float(eigsh(
        laplacian, k=1, which="LM", return_eigenvectors=False
    )[0])
    phi = largest_eigenvalue / 2.0
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError(f"Invalid Laplacian spectral scale: {phi}")
    coefficients = chebyshev_coefficients(
        phi, args.diffusion_time, args.chebyshev_order
    )

    base_mass = np.full(n_cells, 1.0 / n_cells, dtype=np.float64)
    base_diffusion = expm_multiply(
        laplacian,
        base_mass,
        phi,
        coefficients,
        args.chebyshev_order,
    ).reshape(-1)
    base_diffusion = np.maximum(base_diffusion, args.epsilon)

    m_v = np.zeros((n_cells, n_populations), dtype=np.float64)
    sum_m_v = np.zeros(n_populations, dtype=np.float64)
    for condition in range(n_populations):
        condition_rows = rows_per_condition[condition]
        condition_mass = mass_per_condition[condition]
        ratio = condition_mass / base_diffusion[condition_rows]
        m_v[condition_rows, condition] = ratio
        clipped = np.clip(ratio, args.epsilon, 1e32)
        sum_m_v[condition] = float((np.log(clipped + args.epsilon) * condition_mass).sum())
    m_v *= base_mass[0]

    n_pairs = n_populations * (n_populations - 1) // 2
    pair_indices = np.full((n_populations, n_populations), -1, dtype=np.int64)
    pair_index = 0
    for condition_j in range(n_populations - 1):
        for condition_i in range(condition_j + 1, n_populations):
            pair_indices[condition_i, condition_j] = pair_index
            pair_index += 1

    print("Computing condition-wise heat diffusion", flush=True)
    per_condition = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(process_condition)(
            condition_j,
            laplacian,
            m_v[:, condition_j].copy(),
            phi,
            coefficients,
            args.chebyshev_order,
            rows_per_condition,
            mass_per_condition,
            pair_indices,
            args.epsilon,
        )
        for condition_j in range(n_populations - 1)
    )
    sum_m_w = np.zeros(n_pairs, dtype=np.float64)
    for condition_results in per_condition:
        for index, contribution in condition_results:
            sum_m_w[index] = contribution

    scale = 4.0 * base_mass[0] * args.diffusion_time
    sum_m_v *= scale
    sum_m_w *= scale
    sum_m_v_expanded = np.repeat(
        sum_m_v, np.arange(n_populations - 1, -1, -1)
    )
    distance_vector = np.maximum(sum_m_w + sum_m_v_expanded, 0.0)
    distances = np.zeros((n_populations, n_populations), dtype=np.float64)
    distances[np.triu_indices(n_populations, 1)] = distance_vector
    distances += distances.T
    if not np.isfinite(distances).all():
        raise ValueError("Pheno-GS produced non-finite distances")

    output_npy = args.out_dir / "pheno_gs_distances.npy"
    np.save(output_npy, distances.astype(np.float32))
    pd.DataFrame(distances, index=names, columns=names).to_csv(
        args.out_dir / "pheno_gs_distances.csv"
    )
    upper = distances[np.triu_indices(n_populations, 1)]
    summary = {
        "method": "Pheno-GS heat-diffusion population distance",
        "source": "Adapted from user-provided Pheno-GS_BMS.py",
        "cache": str(args.cache),
        "n_populations": n_populations,
        "n_landmarks": n_cells,
        "n_features": int(cells.shape[1]),
        "uniform_landmark_mass": True,
        "diffusion_time": args.diffusion_time,
        "chebyshev_order": args.chebyshev_order,
        "knn": args.knn,
        "laplacian": "combinatorial",
        "largest_eigenvalue": largest_eigenvalue,
        "distance_mean": float(upper.mean()),
        "distance_median": float(np.median(upper)),
        "distance_min": float(upper.min()),
        "distance_max": float(upper.max()),
        "output": str(output_npy),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
