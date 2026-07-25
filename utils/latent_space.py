from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import phate
from matplotlib.lines import Line2D


def categories_from_cache(
    cache_path: str | Path | None,
    color_by: str,
    fallback_labels: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Resolve plotting categories from labels or population-cache metadata."""
    if color_by.lower() == "label" or cache_path is None:
        return np.asarray(fallback_labels).astype(str), "Label"

    with np.load(cache_path, allow_pickle=True) as cached:
        if color_by in cached.files:
            values = np.asarray(cached[color_by])
            if values.ndim == 1 and len(values) == len(fallback_labels):
                return values.astype(str), color_by

        if "group_keys" not in cached.files or "group_names" not in cached.files:
            raise ValueError(
                f"Cannot color by {color_by!r}: cache has no matching array or group metadata"
            )
        group_keys = [str(value) for value in cached["group_keys"]]
        matches = [i for i, key in enumerate(group_keys) if key.lower() == color_by.lower()]
        if not matches:
            raise ValueError(f"Unknown color field {color_by!r}; available fields: {group_keys}")
        field_index = matches[0]
        values = []
        for group_name in cached["group_names"]:
            parts = str(group_name).split("__")
            if field_index >= len(parts):
                raise ValueError(f"Malformed group name in cache: {group_name!r}")
            values.append(parts[field_index])
    return np.asarray(values), group_keys[field_index]


def plot_phate(
    latents: np.ndarray,
    categories: np.ndarray,
    output_path: str | Path,
    category_name: str = "Label",
    knn: int = 30,
    seed: int = 0,
    n_jobs: int = 1,
) -> np.ndarray:
    """Embed HiPoNet latents with PHATE and save a categorical scatter plot."""
    latents = np.asarray(latents, dtype=np.float64)
    categories = np.asarray(categories).astype(str)
    if latents.ndim != 2 or len(latents) < 3:
        raise ValueError("PHATE requires a 2-D latent array with at least three populations")
    if not np.isfinite(latents).all():
        raise ValueError("Latent representations contain non-finite values")
    if len(categories) != len(latents):
        raise ValueError("Plot categories must match the number of latent representations")

    operator = phate.PHATE(
        n_components=2,
        knn=max(2, min(knn, len(latents) - 2)),
        decay=40,
        t="auto",
        gamma=1,
        mds_solver="sgd",
        random_state=seed,
        n_jobs=n_jobs,
        verbose=1,
    )
    embedding = operator.fit_transform(latents)

    category_order = sorted(np.unique(categories))
    counts = Counter(categories)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(category_order), 2)))

    fig, ax = plt.subplots(figsize=(12, 8.4), dpi=220)
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")
    for category, color in zip(category_order, colors):
        mask = categories == category
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.3,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=f"{category} (n={counts[category]})",
        )
        for category, color in zip(category_order, colors)
    ]
    legend_columns = 1 if len(handles) <= 14 else 2
    ax.legend(
        handles=handles,
        title=category_name,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        ncol=legend_columns,
        labelspacing=0.5,
    )
    ax.set_title("HiPoNet Population Latent Space", fontsize=18, fontweight="bold", loc="left")
    ax.text(
        0,
        1.015,
        f"PHATE embedding of {len(latents):,} populations from a {latents.shape[1]}D latent representation",
        transform=ax.transAxes,
        fontsize=10,
        color="#68717A",
    )
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    ax.grid(color="#DCE1E5", linewidth=0.65, alpha=0.58)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B8C0C7")
    fig.subplots_adjust(left=0.08, right=0.78, top=0.88, bottom=0.1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return embedding.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exported HiPoNet latents with PHATE")
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--population_cache", type=Path)
    parser.add_argument("--color_by", default="label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--knn", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    args = parser.parse_args()

    latents = np.load(args.latents)
    if args.labels is not None:
        labels = np.load(args.labels, allow_pickle=True)
    else:
        labels = np.arange(len(latents))
    categories, category_name = categories_from_cache(
        args.population_cache,
        args.color_by,
        labels,
    )
    embedding = plot_phate(
        latents,
        categories,
        args.output,
        category_name=category_name,
        knn=args.knn,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    np.save(args.output.with_name(f"{args.output.stem}_embedding.npy"), embedding)
    print(f"PHATE plot saved to: {args.output}")


if __name__ == "__main__":
    main()
