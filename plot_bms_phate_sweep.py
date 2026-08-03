#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import phate
from matplotlib.lines import Line2D


TREATMENT_ORDER = [
    "DMSO", "H2O", "AH", "C", "CF", "CS", "CSF",
    "F", "L", "O", "S", "SF", "V", "VS",
]
TREATMENT_COLORS = {
    "DMSO": "#20252a", "H2O": "#56B4E9", "AH": "#0072B2",
    "C": "#CC79A7", "CF": "#D55E00", "CS": "#E15759",
    "CSF": "#009E73", "F": "#E69F00", "L": "#59A14F",
    "O": "#B07AA1", "S": "#76B7B2", "SF": "#EDC948",
    "V": "#8C564B", "VS": "#F28E2B",
}


def scatter_categories(axis, embedding, values, order, colors, labels=None):
    counts = Counter(values)
    for value in order:
        mask = values == value
        if np.any(mask):
            axis.scatter(
                embedding[mask, 0], embedding[mask, 1], s=34,
                color=colors[value], alpha=0.8, edgecolors="white", linewidths=0.35,
            )
    return [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor=colors[value], markeredgecolor="white",
            label=f"{labels.get(value, value) if labels else value} (n={counts[value]})",
        )
        for value in order if counts[value]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact_dir", type=Path, required=True)
    parser.add_argument("--phate_knn", type=int, required=True)
    parser.add_argument("--simplicial_k", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1302)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.cache, allow_pickle=True)
    keys = [str(value) for value in cache["group_keys"]]
    rows = [str(value).split("__") for value in cache["group_names"]]
    metadata = {key: np.asarray([row[i] for row in rows]) for i, key in enumerate(keys)}
    latents = np.load(args.latents).astype(np.float64)
    if latents.shape != (len(rows), args.latent_dim):
        raise ValueError(f"Expected {(len(rows), args.latent_dim)}, found {latents.shape}")

    operator = phate.PHATE(
        n_components=2, knn=args.phate_knn, decay=40, t="auto", gamma=1,
        mds_solver="sgd", random_state=args.seed,
        n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), verbose=1,
    )
    embedding = operator.fit_transform(latents)
    treatments = metadata["Treatment"]
    patients = metadata["Patient"]
    cultures = metadata["Culture"]
    patient_order = sorted(set(patients))
    patient_colors = {value: "#62C370" for value in patient_order}
    culture_order = sorted(set(cultures))
    culture_colors = {value: "#009E3B" for value in culture_order}

    figure, axes = plt.subplots(1, 3, figsize=(18, 7.1), dpi=180)
    for axis in axes:
        axis.set_xlabel("PHATE 1")
        axis.set_ylabel("PHATE 2")
        axis.grid(color="#d9dde1", linewidth=0.55, alpha=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    handles = [
        scatter_categories(axes[0], embedding, treatments, TREATMENT_ORDER, TREATMENT_COLORS),
        scatter_categories(axes[1], embedding, patients, patient_order, patient_colors),
        scatter_categories(axes[2], embedding, cultures, culture_order, culture_colors),
    ]
    for axis, title, panel, panel_handles in zip(
        axes, ["Treatment", "Patient", "Microenvironment"], ["a", "b", "c"], handles
    ):
        axis.set_title(title, fontsize=15, weight="bold", pad=12)
        axis.text(-0.12, 1.04, panel, transform=axis.transAxes, fontsize=17, weight="bold")
        axis.legend(
            handles=panel_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
            ncol=3 if title == "Treatment" else 1, frameon=False, fontsize=7,
        )

    n_cells = int(np.asarray(cache["population_sizes"]).sum())
    n_features = len(cache["feature_columns"])
    laplacian = "geometric Hodge Laplacian" if args.simplicial_k == 2 else "graph Laplacian"
    figure.suptitle("HiPoNet BMS All-Cell Population Latent Space", fontsize=19, weight="bold", y=0.98)
    figure.text(
        0.5, 0.935,
        f"{len(rows)} populations | {n_cells:,} cells | {n_features} features | no landmarks | "
        f"one view | latent dimension {args.latent_dim} | K={args.simplicial_k} {laplacian} | "
        f"PHATE knn={args.phate_knn}",
        ha="center", fontsize=10, color="#596169",
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.87, bottom=0.27, wspace=0.25)
    figure.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    np.savez_compressed(
        args.artifact_dir / "phate_embedding.npz", embedding=embedding.astype(np.float32),
        treatments=treatments, patients=patients, cultures=cultures,
    )
    summary = {
        "source_latents": str(args.latents), "latent_dim": args.latent_dim,
        "simplicial_k": args.simplicial_k, "phate_knn": args.phate_knn,
        "n_populations": len(rows), "n_cells": n_cells, "n_features": n_features,
        "representation": "all cells; no landmarks", "output": str(args.output),
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
