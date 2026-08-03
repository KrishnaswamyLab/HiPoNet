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

from utils.latent_space import categories_from_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a three-panel PHATE plot for Mendeley HiPoNet latents"
    )
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phate_knn", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--simplicial_k", type=int, choices=[1, 2], required=True)
    parser.add_argument("--seed", type=int, default=1302)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument(
        "--color_fields",
        nargs=3,
        default=["Treatment", "Patient", "Microenvironment"],
        metavar=("TREATMENT", "PATIENT", "MICROENVIRONMENT"),
    )
    args = parser.parse_args()

    latents = np.load(args.latents)
    if latents.ndim != 2 or latents.shape[1] != args.latent_dim:
        raise ValueError(
            f"Expected latent shape (n, {args.latent_dim}), found {latents.shape}"
        )
    if len(latents) < 3 or not np.isfinite(latents).all():
        raise ValueError("Latents must contain at least three finite populations")

    operator = phate.PHATE(
        n_components=2,
        knn=max(2, min(args.phate_knn, len(latents) - 2)),
        decay=40,
        t="auto",
        gamma=1,
        mds_solver="sgd",
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=1,
    )
    embedding = operator.fit_transform(latents)
    fallback_labels = np.arange(len(latents))
    panel_data = [
        categories_from_cache(args.cache, field, fallback_labels)
        for field in args.color_fields
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.2), dpi=220)
    fig.patch.set_facecolor("#FAFBFC")
    for ax, (categories, category_name) in zip(axes, panel_data):
        ax.set_facecolor("#FAFBFC")
        order = sorted(np.unique(categories))
        counts = Counter(categories)
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(order), 2)))
        for category, color in zip(order, colors):
            selected = categories == category
            ax.scatter(
                embedding[selected, 0],
                embedding[selected, 1],
                s=13,
                color=color,
                alpha=0.72,
                edgecolors="white",
                linewidths=0.25,
            )
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=6,
                label=f"{category} (n={counts[category]})",
            )
            for category, color in zip(order, colors)
        ]
        ax.legend(
            handles=handles,
            title=category_name,
            frameon=False,
            fontsize=7,
            title_fontsize=9,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=2 if len(handles) > 12 else 1,
        )
        ax.set_title(category_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("PHATE 1")
        ax.set_ylabel("PHATE 2")
        ax.grid(color="#DCE1E5", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#B8C0C7")

    laplacian = "geometric" if args.simplicial_k == 2 else "graph"
    fig.suptitle(
        "Mendeley HiPoNet Population Latent Space",
        fontsize=19,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        1.005,
        (
            f"{len(latents):,} populations | latent dim {args.latent_dim} | "
            f"K={args.simplicial_k} ({laplacian}) | PHATE knn={args.phate_knn}"
        ),
        ha="center",
        fontsize=10,
        color="#68717A",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    artifact_dir = args.output.parent.parent / "phate_artifacts" / args.output.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "phate_embedding.npy", embedding.astype(np.float32))
    print(f"PHATE plot saved to: {args.output}")


if __name__ == "__main__":
    main()
