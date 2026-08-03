#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "pHH3", "IdU", "pPDK1", "Geminin", "pMEK1_2", "pNDRG",
    "pMKK4_SEK1", "pBTK", "pSRC", "p4EBP1", "pRB", "pAKT308",
    "pCREB", "pSMAD1_5_9", "pAKT473", "pNF_kB", "pMKK3_MKK6",
    "pP38", "pMAPKAPK", "pAMPKa", "pBAD", "pHistone_H2A", "p90RSK",
    "pP120_catenin", "Beta_catenin_active", "pGSK", "pERK1_2",
    "pSMAD2_3", "PLK", "pDNAPK", "pS6", "cPARP", "pCHK1", "Cyclin_B1",
]
GROUP_KEYS = (
    "Patient", "Culture", "Treatment", "Concentration", "Replicate", "Cell_type"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=Path, required=True)
    parser.add_argument(
        "--member", default="Pheno-GS_BMS/labelled_trellis.csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as archive:
        with archive.open(args.member) as handle:
            dataframe = pd.read_csv(handle)

    missing = [
        column for column in (*GROUP_KEYS, *FEATURE_COLUMNS)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # The source Treatment values include replicate suffixes (for example L_A).
    # Replicate is already a separate grouping field, so retain only treatment type.
    dataframe = dataframe.copy()
    dataframe["Treatment"] = (
        dataframe["Treatment"].astype(str).str.rsplit("_", n=1).str[0]
    )
    finite = np.isfinite(dataframe[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).all(axis=1)
    dataframe = dataframe.loc[finite].reset_index(drop=True)
    if dataframe.empty:
        raise ValueError("No rows have finite values for all selected features")

    scaler = StandardScaler().fit(dataframe[FEATURE_COLUMNS])
    scaled = scaler.transform(dataframe[FEATURE_COLUMNS]).astype(np.float32)
    grouped = dataframe.groupby(list(GROUP_KEYS), sort=True, observed=True).indices

    populations = np.empty(len(grouped), dtype=object)
    group_names = []
    sizes = []
    for population_index, (key, indices) in enumerate(grouped.items()):
        indices = np.asarray(indices, dtype=np.int64)
        populations[population_index] = scaled[indices]
        group_names.append("__".join(map(str, key)))
        sizes.append(len(indices))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        populations=populations,
        labels=np.arange(len(populations), dtype=np.int64),
        num_labels=np.asarray([len(populations)], dtype=np.int64),
        group_names=np.asarray(group_names, dtype=object),
        group_keys=np.asarray(GROUP_KEYS, dtype=object),
        feature_columns=np.asarray(FEATURE_COLUMNS, dtype=object),
        population_sizes=np.asarray(sizes, dtype=np.int64),
        original_population_sizes=np.asarray(sizes, dtype=np.int64),
        landmark_cells_per_group=np.asarray([-1], dtype=np.int64),
        feature_scaler_mean=scaler.mean_.astype(np.float32),
        feature_scaler_scale=scaler.scale_.astype(np.float32),
    )

    summary = {
        "source_zip": str(args.zip_path),
        "source_member": args.member,
        "representation": "all cells; no landmarks or subsampling",
        "n_cells": int(sum(sizes)),
        "n_populations": len(populations),
        "n_features": len(FEATURE_COLUMNS),
        "population_size_min": int(min(sizes)),
        "population_size_median": float(np.median(sizes)),
        "population_size_max": int(max(sizes)),
        "group_keys": list(GROUP_KEYS),
        "feature_columns": FEATURE_COLUMNS,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
