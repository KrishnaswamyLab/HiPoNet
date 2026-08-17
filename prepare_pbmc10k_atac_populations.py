#!/usr/bin/env python
"""Create all-cell PBMC10k ATAC populations from 10x Multiome outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def read_peak_matrix(path: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(value) for value in matrix["shape"][:])
        feature_types = np.asarray(matrix["features"]["feature_type"][:]).astype(str)
        peak_mask = feature_types == "Peaks"
        peak_names = np.asarray(matrix["features"]["name"][:]).astype(str)[peak_mask]
        barcodes = np.asarray(matrix["barcodes"][:]).astype(str)
        features_by_cells = sparse.csc_matrix(
            (
                np.asarray(matrix["data"][:], dtype=np.float32),
                np.asarray(matrix["indices"][:], dtype=np.int32),
                np.asarray(matrix["indptr"][:], dtype=np.int64),
            ),
            shape=shape,
        )
    cells_by_peaks = features_by_cells[peak_mask].T.tocsr()
    cells_by_peaks.data[:] = 1.0
    cells_by_peaks.eliminate_zeros()
    return cells_by_peaks, barcodes, peak_names


def tfidf_lsi(matrix: sparse.csr_matrix, n_components: int, seed: int) -> np.ndarray:
    term_frequency = normalize(matrix, norm="l1", axis=1, copy=True)
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    inverse_document_frequency = np.log1p(matrix.shape[0] / (1.0 + document_frequency))
    tfidf = term_frequency.multiply(inverse_document_frequency).tocsr()
    tfidf = normalize(tfidf, norm="l2", axis=1, copy=False)
    model = TruncatedSVD(
        n_components=n_components + 1,
        algorithm="randomized",
        n_iter=7,
        random_state=seed,
    )
    embedding = model.fit_transform(tfidf)[:, 1 : n_components + 1].astype(np.float32)
    mean = embedding.mean(axis=0, keepdims=True)
    std = embedding.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((embedding - mean) / std).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--replicates_per_cluster", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1302)
    args = parser.parse_args()

    cells_by_peaks, barcodes, peak_names = read_peak_matrix(args.matrix)
    clusters = pd.read_csv(args.clusters)
    cluster_by_barcode = dict(zip(clusters["Barcode"].astype(str), clusters["Cluster"]))
    missing = [barcode for barcode in barcodes if barcode not in cluster_by_barcode]
    if missing:
        raise ValueError(f"Missing cluster assignments for {len(missing)} barcodes")
    cell_clusters = np.asarray([cluster_by_barcode[barcode] for barcode in barcodes], dtype=np.int64)
    embedding = tfidf_lsi(cells_by_peaks, args.n_components, args.seed)

    rng = np.random.default_rng(args.seed)
    populations = []
    population_barcodes = []
    population_clusters = []
    pseudo_replicates = []
    group_names = []
    for cluster in sorted(np.unique(cell_clusters)):
        indices = np.flatnonzero(cell_clusters == cluster)
        rng.shuffle(indices)
        if len(indices) < args.replicates_per_cluster:
            raise ValueError(f"Cluster {cluster} has too few cells for requested replicates")
        for replicate, subset in enumerate(np.array_split(indices, args.replicates_per_cluster), start=1):
            populations.append(embedding[subset])
            population_barcodes.append(barcodes[subset])
            population_clusters.append(str(cluster))
            pseudo_replicates.append(replicate)
            group_names.append(f"ATAC_cluster_{cluster}__replicate_{replicate}")

    population_array = np.empty(len(populations), dtype=object)
    barcode_array = np.empty(len(populations), dtype=object)
    population_array[:] = populations
    barcode_array[:] = population_barcodes
    sizes = np.asarray([len(population) for population in populations], dtype=np.int64)
    labels = np.asarray([int(cluster) - 1 for cluster in population_clusters], dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        populations=population_array,
        labels=labels,
        num_labels=np.asarray([len(np.unique(cell_clusters))], dtype=np.int64),
        group_names=np.asarray(group_names, dtype=object),
        group_keys=np.asarray(["atac_cluster", "pseudo_replicate"], dtype=object),
        atac_cluster=np.asarray(population_clusters, dtype=object),
        pseudo_replicate=np.asarray(pseudo_replicates, dtype=np.int64),
        population_sizes=sizes,
        population_barcodes=barcode_array,
        feature_names=np.asarray([f"LSI{i + 1}" for i in range(args.n_components)], dtype=object),
        feature_source=np.asarray(["ATAC binary TF-IDF/LSI, first component removed"], dtype=object),
        source_matrix=np.asarray([str(args.matrix)], dtype=object),
        source_clusters=np.asarray([str(args.clusters)], dtype=object),
        n_input_peaks=np.asarray([len(peak_names)], dtype=np.int64),
    )
    summary = {
        "source_matrix": str(args.matrix),
        "source_clusters": str(args.clusters),
        "n_cells": int(len(barcodes)),
        "n_input_peaks": int(len(peak_names)),
        "n_atac_clusters": int(len(np.unique(cell_clusters))),
        "replicates_per_cluster": args.replicates_per_cluster,
        "n_populations": int(len(populations)),
        "n_features": args.n_components,
        "population_min": int(sizes.min()),
        "population_median": float(np.median(sizes)),
        "population_max": int(sizes.max()),
        "all_cells_used_once": int(sizes.sum()) == len(barcodes),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
