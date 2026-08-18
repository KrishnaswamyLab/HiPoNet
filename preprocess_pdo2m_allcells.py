#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.preprocessing import StandardScaler


GROUP_KEYS = (
    "Patient",
    "Culture",
    "Treatment",
    "Concentration",
    "Replicate",
    "Cell_type",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scaler_holdout_patients", default="75,99")
    args = parser.parse_args()

    print(f"Loading {args.h5ad}", flush=True)
    adata = ad.read_h5ad(args.h5ad)
    missing = [key for key in GROUP_KEYS if key not in adata.obs]
    if missing:
        raise ValueError(f"Missing population metadata columns: {missing}")

    values = np.asarray(adata.X, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected dense cells-by-features matrix, found {values.shape}")
    feature_columns = [str(value) for value in adata.var_names]
    holdouts = {
        value.strip()
        for value in args.scaler_holdout_patients.split(",")
        if value.strip()
    }
    patients = adata.obs["Patient"].astype(str).to_numpy()
    scaler_mask = ~np.isin(patients, sorted(holdouts))
    if not np.any(scaler_mask):
        raise ValueError("Scaler holdout excludes every cell")

    scaler = StandardScaler().fit(values[scaler_mask])
    scaled = scaler.transform(values).astype(np.float32)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    grouped = adata.obs.groupby(list(GROUP_KEYS), sort=True, observed=True).indices
    populations = np.empty(len(grouped), dtype=object)
    names = []
    sizes = []
    for population_index, (key, indices) in enumerate(grouped.items()):
        indices = np.asarray(indices, dtype=np.int64)
        populations[population_index] = scaled[indices]
        names.append("__".join(map(str, key)))
        sizes.append(len(indices))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        populations=populations,
        labels=np.arange(len(populations), dtype=np.int64),
        num_labels=np.asarray([len(populations)], dtype=np.int64),
        group_names=np.asarray(names, dtype=object),
        group_keys=np.asarray(GROUP_KEYS, dtype=object),
        feature_columns=np.asarray(feature_columns, dtype=object),
        population_sizes=np.asarray(sizes, dtype=np.int64),
        original_population_sizes=np.asarray(sizes, dtype=np.int64),
        landmark_cells_per_group=np.asarray([-1], dtype=np.int64),
        feature_scaler_mean=scaler.mean_.astype(np.float32),
        feature_scaler_scale=scaler.scale_.astype(np.float32),
        scaler_holdout_patients=np.asarray(sorted(holdouts), dtype=object),
    )

    summary = {
        "source_h5ad": str(args.h5ad),
        "representation": "all cells; no landmarks or subsampling",
        "n_cells": int(sum(sizes)),
        "n_populations": len(populations),
        "n_features": len(feature_columns),
        "population_size_min": int(min(sizes)),
        "population_size_median": float(np.median(sizes)),
        "population_size_max": int(max(sizes)),
        "scaler_holdout_patients": sorted(holdouts),
        "group_keys": list(GROUP_KEYS),
        "feature_columns": feature_columns,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
