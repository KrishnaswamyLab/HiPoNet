#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=Path, required=True)
    parser.add_argument("--member", default="Metadata_final_paper")
    parser.add_argument("--population_cache", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(args.population_cache, allow_pickle=True)
    names = [str(value) for value in cache["group_names"]]
    group_keys = [str(value) for value in cache["group_keys"]]
    features = [str(value) for value in cache["feature_columns"]]
    expected_sizes = cache["original_population_sizes"].astype(np.int64)
    scaler_mean = cache["feature_scaler_mean"].astype(np.float32)
    scaler_scale = cache["feature_scaler_scale"].astype(np.float32)

    print(f"Loading {args.zip_path}:{args.member}", flush=True)
    with zipfile.ZipFile(args.zip_path) as archive:
        with archive.open(args.member) as handle:
            frame = pd.read_pickle(handle)
    print(f"Loaded dataframe {frame.shape}", flush=True)

    grouped = frame.groupby(group_keys, sort=False, observed=True).indices
    lookup = {
        "__".join(map(str, key if isinstance(key, tuple) else (key,))): np.asarray(indices)
        for key, indices in grouped.items()
    }
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"Missing {len(missing)} cached populations in source dataframe")
    observed_sizes = np.asarray([len(lookup[name]) for name in names], dtype=np.int64)
    if not np.array_equal(observed_sizes, expected_sizes):
        raise ValueError("Source population sizes do not match the HiPoNet cache")

    offsets = np.concatenate([[0], np.cumsum(observed_sizes)]).astype(np.int64)
    cells_path = args.out_dir / "cells.npy"
    cells = np.lib.format.open_memmap(
        cells_path, mode="w+", dtype=np.float32,
        shape=(int(offsets[-1]), len(features)),
    )
    for population_id, name in enumerate(names):
        values = frame.iloc[lookup[name]][features].to_numpy(dtype=np.float32, copy=True)
        values = (values - scaler_mean) / scaler_scale
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        cells[offsets[population_id] : offsets[population_id + 1]] = values
        if (population_id + 1) % 100 == 0 or population_id + 1 == len(names):
            cells.flush()
            print(f"Wrote populations: {population_id + 1}/{len(names)}", flush=True)
    cells.flush()

    np.save(args.out_dir / "population_offsets.npy", offsets)
    np.save(args.out_dir / "group_names.npy", np.asarray(names, dtype=object))
    np.save(args.out_dir / "feature_columns.npy", np.asarray(features, dtype=object))
    summary = {
        "source_zip": str(args.zip_path),
        "source_member": args.member,
        "population_cache": str(args.population_cache),
        "n_cells": int(offsets[-1]),
        "n_populations": len(names),
        "n_features": len(features),
        "cells": str(cells_path),
        "offsets": str(args.out_dir / "population_offsets.npy"),
        "uses_training_only_scaler_from_cache": True,
        "missing_value_strategy": "training-feature mean after standardization",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / ".complete").touch()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
