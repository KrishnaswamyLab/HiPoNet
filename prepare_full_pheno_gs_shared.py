#!/usr/bin/env python
"""Compute Pheno-GS distances on one shared all-cell Gaussian graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigsh
from scipy.special import ive


def heat_action(laplacian, values, phi, coefficients, order):
    """Chebyshev approximation of exp(-tL) applied to one vector."""
    t0 = values
    result = 0.5 * coefficients[0] * t0
    t1 = (laplacian @ values) / phi - t0
    result = result + coefficients[1] * t1
    for degree in range(2, order + 1):
        t2 = 2.0 * (laplacian @ t1) / phi - 2.0 * t1 - t0
        result = result + coefficients[degree] * t2
        t0, t1 = t1, t2
    return np.asarray(result).reshape(-1)


def load_populations(cache_path):
    with np.load(cache_path, allow_pickle=True) as cache:
        populations = [np.asarray(pop, dtype=np.float32) for pop in cache["populations"]]
    if not populations:
        raise ValueError("Population cache is empty")
    feature_counts = {pop.shape[1] for pop in populations}
    if len(feature_counts) != 1:
        raise ValueError(f"Inconsistent feature counts: {sorted(feature_counts)}")
    return populations


def low_memory_graphtools_laplacian(
    cells,
    knn,
    search_neighbors,
    decay,
    threshold,
    n_jobs,
    seed,
):
    """Reproduce the graphtools adaptive kernel with PyNNDescent neighbors."""
    from pynndescent import NNDescent

    n_neighbors = min(max(search_neighbors, knn + 1), len(cells))
    index = NNDescent(
        cells,
        metric="euclidean",
        n_neighbors=n_neighbors,
        random_state=seed,
        low_memory=True,
        compressed=True,
        n_jobs=n_jobs,
        verbose=True,
    )
    indices, distances = index.neighbor_graph
    indices = np.asarray(indices, dtype=np.int32)
    distances = np.asarray(distances, dtype=np.float32)
    bandwidth = np.maximum(
        distances[:, min(knn, distances.shape[1] - 1)],
        np.finfo(np.float32).eps,
    )
    scaled = distances / bandwidth[:, None]
    weights = np.exp(-np.power(scaled, decay)).astype(np.float32)
    weights[weights < threshold] = 0.0
    indptr = np.arange(
        0,
        (len(cells) + 1) * n_neighbors,
        n_neighbors,
        dtype=np.int64,
    )
    adjacency = sparse.csr_matrix(
        (weights.reshape(-1), indices.reshape(-1), indptr),
        shape=(len(cells), len(cells)),
    )
    adjacency.eliminate_zeros()
    adjacency = ((adjacency + adjacency.T) * 0.5).tocsr()
    adjacency.eliminate_zeros()
    return sparse.csgraph.laplacian(adjacency, normed=False).tocsr()


def prepare(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    populations = load_populations(args.cache)
    sizes = np.asarray([len(pop) for pop in populations], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    cells = np.concatenate(populations, axis=0)
    print(
        f"Building one shared Gaussian graph for {len(populations)} populations, "
        f"{len(cells)} cells, and {cells.shape[1]} features",
        flush=True,
    )
    laplacian = low_memory_graphtools_laplacian(
        cells,
        knn=args.cell_knn,
        search_neighbors=args.search_neighbors,
        decay=args.decay,
        threshold=args.kernel_threshold,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    laplacian = laplacian.astype(np.float32)
    save_npz(args.out_dir / "shared_laplacian.npz", laplacian, compressed=True)

    largest = float(eigsh(laplacian, k=1, which="LM", return_eigenvectors=False)[0])
    phi = largest / 2.0
    coefficients = 2.0 * ive(
        np.arange(args.chebyshev_order + 1), -args.diffusion_time * phi
    )
    base_mass = np.full(len(cells), 1.0 / len(cells), dtype=np.float64)
    base_diffusion = heat_action(
        laplacian, base_mass, phi, coefficients, args.chebyshev_order
    )
    base_diffusion = np.maximum(base_diffusion, args.epsilon)
    np.savez_compressed(
        args.out_dir / "shared_graph_metadata.npz",
        sizes=sizes,
        offsets=offsets,
        phi=np.asarray(phi),
        coefficients=coefficients,
    )
    np.save(args.out_dir / "base_diffusion.npy", base_diffusion.astype(np.float32))
    summary = {
        "stage": "prepared",
        "method": "Pheno-GS on one shared all-cell Gaussian graph",
        "cache": str(args.cache),
        "n_populations": len(populations),
        "n_cells": int(len(cells)),
        "n_features": int(cells.shape[1]),
        "cell_knn": args.cell_knn,
        "neighbor_backend": "pynndescent-low-memory",
        "search_neighbors": args.search_neighbors,
        "kernel_decay": args.decay,
        "kernel_threshold": args.kernel_threshold,
        "laplacian_nnz": int(laplacian.nnz),
        "diffusion_time": args.diffusion_time,
        "chebyshev_order": args.chebyshev_order,
        "n_chunks": args.n_chunks,
        "phi": phi,
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def compute(args):
    laplacian = load_npz(args.out_dir / "shared_laplacian.npz").tocsr()
    metadata = np.load(args.out_dir / "shared_graph_metadata.npz")
    sizes = metadata["sizes"]
    offsets = metadata["offsets"]
    phi = float(metadata["phi"])
    coefficients = metadata["coefficients"]
    base_diffusion = np.load(args.out_dir / "base_diffusion.npy", mmap_mode="r")
    n_cells = laplacian.shape[0]
    n_populations = len(sizes)
    source_indices = np.arange(args.chunk_id, n_populations - 1, args.n_chunks)
    rows_out, cols_out, distances_out = [], [], []
    scale = 4.0 * (1.0 / n_cells) * args.diffusion_time

    for source in source_indices:
        start, stop = int(offsets[source]), int(offsets[source + 1])
        source_mass = 1.0 / float(sizes[source])
        source_ratio = source_mass / np.maximum(base_diffusion[start:stop], args.epsilon)
        sum_source = float(
            np.sum(np.log(np.clip(source_ratio, args.epsilon, 1e32) + args.epsilon))
            * source_mass
        )
        source_measure = np.zeros(n_cells, dtype=np.float64)
        source_measure[start:stop] = source_ratio / n_cells
        diffused = heat_action(
            laplacian, source_measure, phi, coefficients, args.chebyshev_order
        )
        diffused = np.maximum(diffused, args.epsilon)

        for target in range(source + 1, n_populations):
            target_start = int(offsets[target])
            target_stop = int(offsets[target + 1])
            target_mass = 1.0 / float(sizes[target])
            target_ratio = target_mass / diffused[target_start:target_stop]
            cross = float(
                np.sum(np.log(np.clip(target_ratio, args.epsilon, 1e32) + args.epsilon))
                * target_mass
            )
            rows_out.append(source)
            cols_out.append(target)
            distances_out.append(max(scale * (sum_source + cross), 0.0))
        print(f"chunk {args.chunk_id}: source {source} complete", flush=True)

    chunks_dir = args.out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output = chunks_dir / f"chunk_{args.chunk_id:04d}.npz"
    distances = np.asarray(distances_out, dtype=np.float32)
    if not np.isfinite(distances).all():
        raise RuntimeError(f"Chunk {args.chunk_id} produced non-finite distances")
    np.savez_compressed(
        output,
        rows=np.asarray(rows_out, dtype=np.int32),
        cols=np.asarray(cols_out, dtype=np.int32),
        distances=distances,
    )
    print(f"saved {len(distances)} distances to {output}", flush=True)


def finalize(args):
    metadata = np.load(args.out_dir / "shared_graph_metadata.npz")
    sizes = metadata["sizes"]
    n_populations = len(sizes)
    distances = np.zeros((n_populations, n_populations), dtype=np.float32)
    seen = 0
    for chunk_id in range(args.n_chunks):
        path = args.out_dir / "chunks" / f"chunk_{chunk_id:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        chunk = np.load(path)
        rows, cols, values = chunk["rows"], chunk["cols"], chunk["distances"]
        distances[rows, cols] = values
        seen += len(values)
    expected = n_populations * (n_populations - 1) // 2
    if seen != expected:
        raise RuntimeError(f"Expected {expected} pair distances, found {seen}")
    distances += distances.T
    if not np.isfinite(distances).all():
        raise RuntimeError("Completed Pheno-GS matrix contains non-finite values")
    output = args.out_dir / "pheno_gs_distances.npy"
    np.save(output, distances)
    upper = distances[np.triu_indices(n_populations, 1)]
    summary = {
        "method": "Pheno-GS on one shared all-cell Gaussian graph",
        "cache": str(args.cache),
        "n_populations": n_populations,
        "n_cells": int(sizes.sum()),
        "n_pairs": int(expected),
        "distance_min": float(upper.min()),
        "distance_median": float(np.median(upper)),
        "distance_mean": float(upper.mean()),
        "distance_max": float(upper.max()),
        "distance_output": str(output),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compute", "finalize"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--cell_knn", type=int, default=10)
    parser.add_argument("--search_neighbors", type=int, default=32)
    parser.add_argument("--decay", type=float, default=40.0)
    parser.add_argument("--kernel_threshold", type=float, default=1e-4)
    parser.add_argument("--diffusion_time", type=float, default=10.0)
    parser.add_argument("--chebyshev_order", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--n_chunks", type=int, default=128)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1302)
    args = parser.parse_args()
    if not 0 <= args.chunk_id < args.n_chunks:
        raise ValueError("--chunk_id must be in [0, n_chunks)")
    {"prepare": prepare, "compute": compute, "finalize": finalize}[args.stage](args)


if __name__ == "__main__":
    main()
