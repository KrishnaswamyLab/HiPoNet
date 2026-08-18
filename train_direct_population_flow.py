#!/usr/bin/env python
"""Train direct latent-conditioned flow matching from Gaussian cells to populations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flow_matching.path import CondOTProbPath
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch import nn

from models.population_flow import ConditionedResidualBlock, SinusoidalTimeEmbedding
from utils.population_generation import (
    aggregate,
    emd_point_cloud_loss,
    evaluation_metrics,
    sample_cells,
)


class LatentConditionedVelocityField(nn.Module):
    """Population-specific velocity field v(x_t, t, z)."""

    def __init__(
        self,
        cell_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.condition = nn.Sequential(
            nn.Linear(time_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cell_input = nn.Linear(cell_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ConditionedResidualBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, cell_dim)

    def forward(
        self, cells: torch.Tensor, time: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        if time.ndim == 0:
            time = time.expand(len(cells))
        elif time.ndim == 2 and time.shape[1] == 1:
            time = time[:, 0]
        if latent.ndim != 2 or latent.shape[0] != len(cells):
            raise ValueError("Latent must have shape (cells, latent_dim)")
        condition = self.condition(
            torch.cat((self.time_embedding(time), latent), dim=1)
        )
        hidden = self.cell_input(cells)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(F.silu(self.output_norm(hidden)))


def integrate_direct_flow(
    model: LatentConditionedVelocityField,
    source: torch.Tensor,
    latent: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """Integrate v(x_t, t, z) from Gaussian source to t=1 with Heun steps."""
    cells = source
    expanded_latent = latent[:, None].expand(-1, cells.shape[1], -1)
    step_size = 1.0 / n_steps
    for step in range(n_steps):
        time = cells.new_full(cells.shape[:2], step / n_steps)
        velocity = model(
            cells.flatten(0, 1),
            time.flatten(),
            expanded_latent.flatten(0, 1),
        ).view_as(cells)
        proposal = cells + step_size * velocity
        next_time = cells.new_full(cells.shape[:2], (step + 1) / n_steps)
        next_velocity = model(
            proposal.flatten(0, 1),
            next_time.flatten(),
            expanded_latent.flatten(0, 1),
        ).view_as(cells)
        cells = cells + 0.5 * step_size * (velocity + next_velocity)
    return cells


def fixed_gaussian(
    population_id: int,
    n_cells: int,
    cell_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + population_id)
    return torch.randn(1, n_cells, cell_dim, device=device, generator=generator)


def exact_emd(prediction: np.ndarray, target: np.ndarray) -> float:
    distances = cdist(prediction, target, metric="euclidean")
    source_indices, target_indices = linear_sum_assignment(distances)
    return float(distances[source_indices, target_indices].mean())


def summarize(rows: list[dict], method: str) -> dict[str, float]:
    selected = [row for row in rows if row["method"] == method]
    result = aggregate(selected)
    emd = np.asarray([row["emd"] for row in selected], dtype=np.float64)
    result.update(
        {
            "emd": float(emd.mean()),
            "emd_standard_deviation": float(
                emd.std(ddof=1) if len(emd) > 1 else 0.0
            ),
            "emd_median": float(np.median(emd)),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_cache", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_patient", default="75")
    parser.add_argument("--test_patient", default="99")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--emd_solver", choices=("exact", "sinkhorn_log"), default="sinkhorn_log"
    )
    parser.add_argument("--emd_regularization", type=float, default=0.2)
    parser.add_argument("--emd_iterations", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=1600)
    parser.add_argument("--integration_steps", type=int, default=32)
    parser.add_argument("--metric_points", type=int, default=256)
    parser.add_argument("--eval_populations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1840)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cache = np.load(args.population_cache, allow_pickle=True)
    populations = [np.asarray(value, dtype=np.float32) for value in cache["populations"]]
    names = [str(value) for value in cache["group_names"]]
    latents = np.asarray(np.load(args.latents), dtype=np.float32)
    if len(latents) != len(populations):
        raise ValueError(
            f"Found {len(populations)} populations but {len(latents)} latent vectors"
        )
    patients = np.asarray([name.split("__", 1)[0] for name in names])
    validation_ids = np.flatnonzero(patients == str(args.validation_patient))
    test_ids = np.flatnonzero(patients == str(args.test_patient))
    train_ids = np.flatnonzero(
        (patients != str(args.validation_patient))
        & (patients != str(args.test_patient))
    )
    if min(len(train_ids), len(validation_ids), len(test_ids)) == 0:
        raise ValueError("Train, validation, and test splits must be nonempty")

    cell_dim = populations[0].shape[1]
    latent_dim = latents.shape[1]
    latent_mean = latents[train_ids].mean(0).astype(np.float32)
    latent_std = np.maximum(latents[train_ids].std(0), 1e-6).astype(np.float32)
    scaled_latents = ((latents - latent_mean) / latent_std).astype(np.float32)
    latent_tensor = torch.from_numpy(scaled_latents).to(device)

    transform_rng = np.random.default_rng(args.seed + 1)
    transform_cells = np.concatenate(
        [
            sample_cells(
                populations[int(index)],
                min(256, len(populations[int(index)])),
                transform_rng,
            )
            for index in train_ids
        ]
    )
    cell_mean = transform_cells.mean(0).astype(np.float32)
    cell_std = np.maximum(transform_cells.std(0), 1e-4).astype(np.float32)

    def to_model(values: np.ndarray) -> np.ndarray:
        return ((values - cell_mean) / cell_std).astype(np.float32)

    def from_model(values: np.ndarray) -> np.ndarray:
        return (values * cell_std + cell_mean).astype(np.float32)

    model = LatentConditionedVelocityField(
        cell_dim, latent_dim, args.hidden_dim, n_blocks=args.n_blocks
    ).to(device)
    nn.init.zeros_(model.output.weight)
    nn.init.zeros_(model.output.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    probability_path = CondOTProbPath()

    def validation() -> float:
        model.eval()
        values = []
        with torch.no_grad():
            for population_id in validation_ids:
                population_id = int(population_id)
                target = torch.from_numpy(to_model(populations[population_id])).to(
                    device
                )[None]
                source = fixed_gaussian(
                    population_id, target.shape[1], cell_dim, args.seed + 10_000, device
                )
                prediction = integrate_direct_flow(
                    model,
                    source,
                    latent_tensor[population_id][None],
                    args.integration_steps,
                )
                value, _ = emd_point_cloud_loss(
                    prediction,
                    target,
                    args.emd_solver,
                    args.emd_regularization,
                    args.emd_iterations,
                )
                values.append(float(value))
        model.train()
        return float(np.mean(values))

    baseline_validation = validation()
    best_validation = baseline_validation
    best_step = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    history = []
    for step in range(1, args.steps + 1):
        population_id = int(rng.choice(train_ids))
        target = torch.from_numpy(to_model(populations[population_id])).to(device)
        source = torch.randn_like(target)
        time = torch.rand(len(target), device=device)
        path_sample = probability_path.sample(x_0=source, x_1=target, t=time)
        latent = latent_tensor[population_id][None].expand(len(target), -1)
        predicted_velocity = model(path_sample.x_t, path_sample.t, latent)
        loss = F.mse_loss(predicted_velocity, path_sample.dx_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            score = validation()
            row = {
                "step": step,
                "population_id": population_id,
                "training_cells": len(target),
                "flow_matching_mse": float(loss.detach()),
                "validation_emd": score,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if score < best_validation:
                best_validation = score
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if step - best_step >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    eval_rng = np.random.default_rng(args.seed + 3)
    selected_test = (
        eval_rng.choice(
            test_ids, size=min(args.eval_populations, len(test_ids)), replace=False
        )
        if args.eval_populations > 0
        else test_ids
    )
    projection_rng = np.random.default_rng(args.seed + 2)
    metric_projections = projection_rng.normal(size=(64, cell_dim))
    metric_projections /= np.linalg.norm(metric_projections, axis=1, keepdims=True)
    rows = []
    with torch.no_grad():
        for population_id in selected_test:
            population_id = int(population_id)
            target = populations[population_id]
            n_cells = len(target)
            source = fixed_gaussian(
                population_id, n_cells, cell_dim, args.seed + 30_000, device
            )
            prediction = integrate_direct_flow(
                model,
                source,
                latent_tensor[population_id][None],
                args.integration_steps,
            )
            generated = {
                "gaussian_source": from_model(source[0].cpu().numpy()),
                "direct_flow_matching": from_model(prediction[0].cpu().numpy()),
                "real_vs_real": sample_cells(target, n_cells, eval_rng),
            }
            for method, values in generated.items():
                count = min(args.metric_points, len(values), len(target))
                prediction_sample = sample_cells(values, count, eval_rng)
                target_sample = sample_cells(target, count, eval_rng)
                rows.append(
                    {
                        "population_id": population_id,
                        "population_name": names[population_id],
                        "true_count": n_cells,
                        "generated_count": len(values),
                        "method": method,
                        **evaluation_metrics(
                            prediction_sample, target_sample, metric_projections
                        ),
                        "emd": exact_emd(prediction_sample, target_sample),
                    }
                )

    with (args.output_dir / "test_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    methods = ("gaussian_source", "direct_flow_matching", "real_vs_real")
    metrics = {method: summarize(rows, method) for method in methods}
    summary = {
        "method": "direct HiPoNet-conditioned Gaussian flow matching",
        "flow_matching_implementation": "Meta flow_matching 1.0.10",
        "source_distribution": "standard Gaussian in normalized 44D cell space",
        "path": "CondOTProbPath linear Gaussian-to-real-cell path",
        "training_pairing": "independent Gaussian and within-population real cells",
        "objective": "pure flow-matching MSE; no decoder or auxiliary losses",
        "checkpoint_selection": "POT EMD between generated and real validation populations",
        "validation_emd_solver": args.emd_solver,
        "validation_emd_regularization": args.emd_regularization,
        "validation_emd_iterations": args.emd_iterations,
        "conditioning": "x_t, t, and the population HiPoNet latent z",
        "cell_dim": cell_dim,
        "latent_dim": latent_dim,
        "generation_count_source": "target_population_count",
        "training_cell_count_source": "all cells in the selected population",
        "validation_populations": int(len(validation_ids)),
        "test_populations": int(len(selected_test)),
        "baseline_validation": baseline_validation,
        "best_step": best_step,
        "best_validation": best_validation,
        "held_out_test_statistics": {
            method: {
                metric: {
                    "mean": metrics[method][metric],
                    "standard_deviation": metrics[method][
                        f"{metric}_standard_deviation"
                    ],
                }
                for metric in ("chamfer", "emd", "pcc", "scc")
            }
            for method in methods
        },
        "distribution_metrics": metrics,
    }
    torch.save(
        {
            "model_state_dict": best_state,
            "summary": summary,
            "cell_mean": cell_mean,
            "cell_std": cell_std,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
        },
        args.output_dir / "direct_population_flow.pt",
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
