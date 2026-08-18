#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


DEFAULT_GROUP_KEYS = (
    "Patient",
    "Culture",
    "Treatment",
    "Concentration",
    "Replicate",
    "Cell_type",
)
METADATA_COLUMNS = {
    "Date",
    "Patient",
    "Culture",
    "Treatment",
    "Concentration",
    "Replicate",
    "Cell_type",
    "Plate",
    "Batch",
}


def read_dataframe(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            return pd.read_pickle(handle)


def infer_feature_columns(df: pd.DataFrame, group_keys: tuple[str, ...], n_features: int | None) -> list[str]:
    excluded = set(METADATA_COLUMNS).union(group_keys)
    numeric = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    if n_features is not None:
        numeric = numeric[:n_features]
    if not numeric:
        raise ValueError("Could not infer numeric marker/PTM feature columns.")
    return numeric


def filter_pdo_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the PDO population definition supplied by the project team."""
    filtered = df[
        df["Culture"].isin(["PDO", "PDOF"])
        & (df["Cell_type"] == "PDOs")
        & ~df["Treatment"].isin(["O", "H2O"])
    ].copy()
    concentrations = pd.to_numeric(filtered["Concentration"], errors="raise")
    maximum = concentrations.groupby(filtered["Treatment"], observed=True).transform(
        "max"
    )
    filtered = filtered[concentrations.eq(maximum)].copy()
    filtered["Condition"] = (
        filtered["Patient"].astype(str)
        + "*"
        + filtered["Treatment"].astype(str)
        + "*"
        + filtered["Concentration"].astype(str)
        + "*"
        + filtered["Culture"].astype(str)
        + "*"
        + filtered["Cell_type"].astype(str)
        + "_"
        + filtered["Replicate"].astype(str)
    )
    return filtered


def make_landmark_cloud(
    values: np.ndarray,
    n_landmarks: int | None,
    n_samples: int | None,
    rng: np.random.Generator,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent one full cell population with centroids fit on all cells."""
    if n_landmarks is not None and n_samples is not None:
        raise ValueError("Choose landmarks or sampled cells, not both")

    if n_samples is not None:
        count = min(n_samples, values.shape[0])
        selected = rng.choice(values.shape[0], size=count, replace=False)
        cloud = values[selected].astype(np.float32)
        weights = np.full(count, 1.0 / count, dtype=np.float32)
        return cloud, weights

    if n_landmarks is None:
        weights = np.full(values.shape[0], 1.0 / max(values.shape[0], 1), dtype=np.float32)
        return values.astype(np.float32), weights

    if values.shape[0] <= n_landmarks:
        weights = np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float32)
        return values.astype(np.float32), weights

    model = MiniBatchKMeans(
        n_clusters=n_landmarks,
        random_state=seed,
        batch_size=max(4096, n_landmarks * 8),
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    assignments = model.fit_predict(values)
    counts = np.bincount(assignments, minlength=n_landmarks).astype(np.float32)

    # Stable order: largest landmarks first, with tiny deterministic jitter only
    # for exact ties so repeated runs with the same seed are reproducible.
    tie_break = rng.uniform(0.0, 1e-6, size=n_landmarks)
    order = np.lexsort((tie_break, -counts))
    centroids = model.cluster_centers_[order].astype(np.float32)
    weights = counts[order]
    weights = weights / max(float(weights.sum()), 1.0)
    return centroids, weights.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=Path, default=Path("data/pdo_caf_mendeley/Metadata_PDO_CAF_screening.zip"))
    parser.add_argument("--member", default="Metadata_final_paper")
    parser.add_argument("--out_dir", type=Path, default=Path("data/pdo_caf_mendeley"))
    parser.add_argument("--n_features", type=int, default=32)
    parser.add_argument("--min_cells_per_group", type=int, default=1)
    parser.add_argument("--max_groups", type=int, default=None)
    parser.add_argument(
        "--max_contexts",
        type=int,
        default=4,
        help="Balance selected groups across this many patient/culture/cell-type contexts.",
    )
    parser.add_argument(
        "--landmark_cells_per_group",
        type=int,
        default=None,
        help=(
            "If set, fit this many MiniBatchKMeans centroids per population using "
            "all cells in the group, and save the centroids as the point cloud."
        ),
    )
    parser.add_argument(
        "--sample_cells_per_group",
        type=int,
        default=None,
        help="Uniformly sample up to this many real cells from every population.",
    )
    parser.add_argument("--group_keys", default=",".join(DEFAULT_GROUP_KEYS))
    parser.add_argument(
        "--pdo_conditions_only",
        action="store_true",
        help=(
            "Keep PDO/PDOF cultures and PDO cells, exclude O/H2O treatments, "
            "retain each treatment's maximum available concentration, and create "
            "the requested Condition label"
        ),
    )
    parser.add_argument(
        "--scaler_holdout_field",
        default=None,
        help="Metadata field whose held-out values are excluded when fitting StandardScaler",
    )
    parser.add_argument(
        "--scaler_holdout_values",
        default="",
        help="Comma-separated metadata values excluded when fitting StandardScaler",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    group_keys = tuple(k.strip() for k in args.group_keys.split(",") if k.strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.zip_path}:{args.member}", flush=True)
    df = read_dataframe(args.zip_path, args.member)
    if args.pdo_conditions_only:
        df = filter_pdo_conditions(df)
        print(
            f"PDO condition filter retained {len(df)} cells and "
            f"{df['Condition'].nunique()} conditions",
            flush=True,
        )
    print(f"Loaded dataframe shape: {df.shape}", flush=True)
    print("Columns:", json.dumps(list(map(str, df.columns))), flush=True)

    missing = [k for k in group_keys if k not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns: {missing}")

    feature_columns = infer_feature_columns(df, group_keys, args.n_features)
    print("Feature columns:", json.dumps(feature_columns), flush=True)

    grouped = df.groupby(list(group_keys), sort=True, observed=True).indices
    eligible = {}
    for key, indices in grouped.items():
        indices = np.asarray(indices)
        if len(indices) >= args.min_cells_per_group:
            normalized_key = tuple(map(str, key if isinstance(key, tuple) else (key,)))
            eligible[normalized_key] = indices
    if not eligible:
        raise ValueError("No eligible groups with enough cells.")

    context_positions = [group_keys.index(key) for key in ("Patient", "Culture", "Cell_type")]
    context_groups: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for key in eligible:
        context = tuple(key[position] for position in context_positions)
        context_groups.setdefault(context, []).append(key)
    ordered_contexts = sorted(
        context_groups,
        key=lambda context: (-len(context_groups[context]), context),
    )
    selected_contexts = (
        ordered_contexts[: args.max_contexts]
        if args.max_contexts is not None and args.max_contexts > 0
        else ordered_contexts
    )
    context_queues = [sorted(context_groups[context]) for context in selected_contexts]
    selected_keys: list[tuple[str, ...]] = []
    while any(context_queues) and (
        args.max_groups is None or len(selected_keys) < args.max_groups
    ):
        for queue in context_queues:
            if queue and (args.max_groups is None or len(selected_keys) < args.max_groups):
                selected_keys.append(queue.pop(0))
    eligible_groups = ["__".join(key) for key in selected_keys]
    condition_fields = (
        "Patient",
        "Treatment",
        "Concentration",
        "Culture",
        "Cell_type",
        "Replicate",
    )
    if all(field in group_keys for field in condition_fields):
        condition_names = []
        for key in selected_keys:
            metadata = dict(zip(group_keys, key))
            condition_names.append(
                "*".join(metadata[field] for field in condition_fields[:-1])
                + "_"
                + metadata["Replicate"]
            )
    else:
        condition_names = eligible_groups
    group_to_indices = {
        "__".join(key): eligible[key]
        for key in selected_keys
    }

    print(f"Eligible groups: {len(eligible_groups)}", flush=True)
    print(f"Selected contexts: {selected_contexts}", flush=True)
    selected_counts = np.asarray([len(group_to_indices[name]) for name in eligible_groups])
    print(pd.Series(selected_counts).describe().to_string(), flush=True)

    selected_rows = np.concatenate([group_to_indices[name] for name in eligible_groups])
    scaler_rows = selected_rows
    scaler_holdout_values = {
        value.strip() for value in args.scaler_holdout_values.split(",") if value.strip()
    }
    if args.scaler_holdout_field is not None:
        if args.scaler_holdout_field not in group_keys:
            raise ValueError(
                f"Scaler holdout field {args.scaler_holdout_field!r} not in {group_keys}"
            )
        holdout_position = group_keys.index(args.scaler_holdout_field)
        scaler_group_names = [
            "__".join(key)
            for key in selected_keys
            if key[holdout_position] not in scaler_holdout_values
        ]
        if not scaler_group_names:
            raise ValueError("Scaler holdout excluded every population")
        scaler_rows = np.concatenate([group_to_indices[name] for name in scaler_group_names])
    scaler = StandardScaler().fit(df.iloc[scaler_rows][feature_columns])

    rng = np.random.default_rng(args.seed)
    populations = []
    landmark_weights = []
    original_population_sizes = []
    labels = []
    for group_i, group_name in enumerate(eligible_groups):
        idx = group_to_indices[group_name]
        rng.shuffle(idx)
        full_values = scaler.transform(df.iloc[idx][feature_columns]).astype(np.float32)
        # Missing channels arise from panel/batch differences. After fitting the
        # scaler on training patients only, zero is the corresponding training mean.
        full_values = np.nan_to_num(full_values, nan=0.0, posinf=0.0, neginf=0.0)
        cloud, weights = make_landmark_cloud(
            full_values,
            args.landmark_cells_per_group,
            args.sample_cells_per_group,
            rng,
            args.seed + group_i,
        )
        populations.append(cloud)
        landmark_weights.append(weights)
        original_population_sizes.append(full_values.shape[0])
        labels.append(group_i)

    populations_obj = np.empty(len(populations), dtype=object)
    landmark_weights_obj = np.empty(len(landmark_weights), dtype=object)
    for i, pop in enumerate(populations):
        populations_obj[i] = pop
        landmark_weights_obj[i] = landmark_weights[i]
    labels = np.asarray(labels, dtype=np.int64)
    if args.landmark_cells_per_group is not None:
        representation = f"landmarks{args.landmark_cells_per_group}"
    elif args.sample_cells_per_group is not None:
        representation = f"sampled{args.sample_cells_per_group}"
    else:
        representation = "allcells"
    cache_name = (
        f"populations_{representation}_features{len(feature_columns)}_"
        f"n{len(populations)}_contexts{len(selected_contexts)}_seed{args.seed}_"
        f"{'-'.join(group_keys)}.npz"
    )
    out_path = args.out_dir / cache_name
    np.savez_compressed(
        out_path,
        populations=populations_obj,
        labels=labels,
        group_names=np.asarray(eligible_groups, dtype=object),
        condition_names=np.asarray(condition_names, dtype=object),
        feature_columns=np.asarray(feature_columns, dtype=object),
        group_keys=np.asarray(group_keys, dtype=object),
        num_labels=np.asarray([len(eligible_groups)], dtype=np.int64),
        population_sizes=np.asarray([pop.shape[0] for pop in populations], dtype=np.int64),
        original_population_sizes=np.asarray(original_population_sizes, dtype=np.int64),
        landmark_weights=landmark_weights_obj,
        landmark_cells_per_group=np.asarray(
            [-1 if args.landmark_cells_per_group is None else args.landmark_cells_per_group],
            dtype=np.int64,
        ),
        sample_cells_per_group=np.asarray(
            [-1 if args.sample_cells_per_group is None else args.sample_cells_per_group],
            dtype=np.int64,
        ),
        feature_scaler_mean=scaler.mean_.astype(np.float32),
        feature_scaler_scale=scaler.scale_.astype(np.float32),
    )
    latest_path = args.out_dir / "populations_expanded_latest.npz"
    shutil.copyfile(out_path, latest_path)
    summary = {
        "source_zip": str(args.zip_path),
        "source_member": args.member,
        "dataframe_shape": list(df.shape),
        "cache": str(out_path),
        "latest_cache": str(latest_path),
        "n_populations": len(populations),
        "population_size_min": int(min(pop.shape[0] for pop in populations)),
        "population_size_median": float(np.median([pop.shape[0] for pop in populations])),
        "population_size_max": int(max(pop.shape[0] for pop in populations)),
        "original_population_size_min": int(min(original_population_sizes)),
        "original_population_size_median": float(np.median(original_population_sizes)),
        "original_population_size_max": int(max(original_population_sizes)),
        "landmark_cells_per_group": args.landmark_cells_per_group,
        "sample_cells_per_group": args.sample_cells_per_group,
        "n_eligible_groups": len(eligible_groups),
        "selected_contexts": [list(context) for context in selected_contexts],
        "group_keys": list(group_keys),
        "pdo_conditions_only": args.pdo_conditions_only,
        "condition_count": len(set(condition_names)),
        "feature_columns": feature_columns,
        "scaler_fit_cell_count": int(len(scaler_rows)),
        "scaler_holdout_field": args.scaler_holdout_field,
        "scaler_holdout_values": sorted(scaler_holdout_values),
        "missing_value_strategy": "training-feature mean after standardization",
        "feature_missing_counts": {
            str(k): int(v) for k, v in df[feature_columns].isna().sum().items()
        },
        "seed": args.seed,
    }
    (args.out_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
