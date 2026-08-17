#!/usr/bin/env python
"""Evaluate a joint population generator with held-out Chamfer and exact EMD."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from models.population_flow import ConditionalPopulationFlow
from train_soft_pointcloud_corrective_flow import integrate_corrective_flow, sample_cells
from train_variable_population_generator import VariableSoftPointCloudDecoder


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    squared = cdist(prediction, target, metric="sqeuclidean")
    chamfer = squared.min(1).mean() + squared.min(0).mean()
    distances = np.sqrt(squared)
    source_indices, target_indices = linear_sum_assignment(distances)
    emd = distances[source_indices, target_indices].mean()
    return {"chamfer": float(chamfer), "emd": float(emd)}


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(array.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(len(array))
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "standard_deviation": standard_deviation,
        "median": float(np.median(array)),
        "ci95_low": float(array.mean() - 1.96 * standard_error),
        "ci95_high": float(array.mean() + 1.96 * standard_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_cache", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--test_patient", default="99")
    parser.add_argument("--eval_populations", type=int, default=48)
    parser.add_argument("--metric_points", type=int, default=256)
    parser.add_argument("--query_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--integration_steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1824)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cache = np.load(args.population_cache, allow_pickle=True)
    populations = [np.asarray(value, dtype=np.float32) for value in cache["populations"]]
    names = [str(value) for value in cache["group_names"]]
    latents = np.asarray(np.load(args.latents), dtype=np.float32)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    summary = checkpoint["summary"]
    cell_dim = int(summary["cell_dim"])
    latent_dim = int(summary["latent_dim"])

    latent_mean = np.asarray(checkpoint["latent_mean"], dtype=np.float32)
    latent_std = np.asarray(checkpoint["latent_std"], dtype=np.float32)
    cell_mean = np.asarray(checkpoint["cell_mean"], dtype=np.float32)
    cell_std = np.asarray(checkpoint["cell_std"], dtype=np.float32)
    latent_tensor = torch.from_numpy(
        ((latents - latent_mean) / latent_std).astype(np.float32)
    ).to(device)

    def from_model(values: np.ndarray) -> np.ndarray:
        return (values * cell_std + cell_mean).astype(np.float32)

    decoder = VariableSoftPointCloudDecoder(
        latent_dim, args.query_dim, cell_dim, args.hidden_dim, 0.10, 1.0
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.eval()
    flow = ConditionalPopulationFlow(
        cell_dim=cell_dim,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
    ).to(device)
    flow.load_state_dict(checkpoint["flow_state_dict"])
    flow.eval()

    patients = np.asarray([name.split("__", 1)[0] for name in names])
    test_ids = np.flatnonzero(patients == str(args.test_patient))
    selected = rng.choice(
        test_ids, size=min(args.eval_populations, len(test_ids)), replace=False
    )
    rows = []
    with torch.no_grad():
        for population_id in selected:
            target = populations[int(population_id)]
            n_cells = len(target)
            latent = latent_tensor[int(population_id)][None]
            queries = torch.randn(1, n_cells, args.query_dim, device=device)
            noise = torch.randn(1, n_cells, cell_dim, device=device)
            source, _, _ = decoder(latent, queries, noise)
            corrected = integrate_corrective_flow(
                flow, source, latent, args.integration_steps
            )
            generated = {
                "joint_soft_decoder": from_model(source[0].cpu().numpy()),
                "joint_corrected_flow": from_model(corrected[0].cpu().numpy()),
                "real_vs_real": sample_cells(target, n_cells, rng),
            }
            for method, prediction in generated.items():
                count = min(args.metric_points, len(prediction), len(target))
                prediction_sample = sample_cells(prediction, count, rng)
                target_sample = sample_cells(target, count, rng)
                rows.append(
                    {
                        "population_id": int(population_id),
                        "population_name": names[int(population_id)],
                        "method": method,
                        "metric_points": count,
                        **metrics(prediction_sample, target_sample),
                    }
                )

    csv_path = args.output_dir / "heldout_chamfer_emd.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    methods = sorted({row["method"] for row in rows})
    result = {
        "test_patient": str(args.test_patient),
        "metric_points": args.metric_points,
        "emd_definition": "mean Euclidean cost of exact uniform-mass assignment",
        "methods": {
            method: {
                metric: summarize(
                    [row[metric] for row in rows if row["method"] == method]
                )
                for metric in ("chamfer", "emd")
            }
            for method in methods
        },
    }
    json_path = args.output_dir / "heldout_chamfer_emd_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
