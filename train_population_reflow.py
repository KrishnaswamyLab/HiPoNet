#!/usr/bin/env python
"""Train a HiPoNet soft point-cloud decoder followed by unconditional reflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch import nn
from flow_matching.path import CondOTProbPath

from models.population_flow import PopulationVelocityField
from utils.population_generation import (
    aggregate,
    distribution_loss,
    evaluation_metrics,
    integrate_corrective_flow,
    sample_cells,
)


class SoftMLPDecoder(nn.Module):
    """Map a population latent and per-cell noise directly to a soft point cloud."""

    def __init__(
        self, latent_dim: int, noise_dim: int, cell_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cell_dim),
        )

    def forward(self, latent: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 3 or noise.shape[0] != latent.shape[0]:
            raise ValueError("Noise must have shape (batch, cells, noise_dim)")
        if noise.shape[2] != self.noise_dim:
            raise ValueError(
                f"Expected noise dimension {self.noise_dim}, found {noise.shape[2]}"
            )
        n_points = noise.shape[1]
        expanded_latent = latent[:, None].expand(-1, n_points, -1)
        return self.mlp(torch.cat((expanded_latent, noise), dim=2))


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


def sample_decoder_noise(
    batch_size: int,
    n_points: int,
    noise_dim: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the Gaussian source used by the soft point-cloud decoder."""
    return torch.randn(
        batch_size,
        n_points,
        noise_dim,
        device=device,
        generator=generator,
    )


def sample_flow_path(
    probability_path: CondOTProbPath,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Meta's original conditional OT probability path per population."""
    times, paths, velocities = [], [], []
    for source_cloud, target_cloud in zip(source, target):
        time = torch.rand(len(source_cloud), device=source.device)
        path_sample = probability_path.sample(
            x_0=source_cloud, x_1=target_cloud, t=time
        )
        times.append(path_sample.t[:, None])
        paths.append(path_sample.x_t)
        velocities.append(path_sample.dx_t)
    return torch.stack(times), torch.stack(paths), torch.stack(velocities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_cache", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_patient", default="75")
    parser.add_argument("--test_patient", default="99")
    parser.add_argument("--decoder_steps", type=int, default=5000)
    parser.add_argument("--flow_steps", type=int, default=4000)
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--decoder_learning_rate", type=float, default=3e-4)
    parser.add_argument("--flow_learning_rate", type=float, default=5e-5)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=1600)
    parser.add_argument("--integration_steps", type=int, default=32)
    parser.add_argument("--metric_points", type=int, default=256)
    parser.add_argument("--eval_populations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1830)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.noise_dim < 1:
        parser.error("--noise_dim must be positive")
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
            "Population/latent count mismatch: "
            f"{len(populations)} populations but {len(latents)} latent vectors"
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
            sample_cells(populations[int(index)], min(256, len(populations[int(index)])), transform_rng)
            for index in train_ids
        ]
    )
    cell_mean = transform_cells.mean(0).astype(np.float32)
    cell_std = np.maximum(transform_cells.std(0), 1e-4).astype(np.float32)

    def to_model(values: np.ndarray) -> np.ndarray:
        return ((values - cell_mean) / cell_std).astype(np.float32)

    def from_model(values: np.ndarray) -> np.ndarray:
        return (values * cell_std + cell_mean).astype(np.float32)

    projection_rng = np.random.default_rng(args.seed + 2)
    projections = projection_rng.normal(size=(32, cell_dim)).astype(np.float32)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    projection_tensor = torch.from_numpy(projections).to(device)

    decoder = SoftMLPDecoder(
        latent_dim, args.noise_dim, cell_dim, args.hidden_dim
    ).to(device)
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.decoder_learning_rate, weight_decay=1e-5
    )
    decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        decoder_optimizer, T_max=args.decoder_steps
    )

    def fixed_noise(population_id: int, n_points: int, offset: int) -> torch.Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + offset + population_id)
        return sample_decoder_noise(
            1, n_points, args.noise_dim, device, generator=generator
        )

    def decoder_validation() -> float:
        decoder.eval()
        values = []
        with torch.no_grad():
            for population_id in validation_ids:
                population_id = int(population_id)
                target = torch.from_numpy(to_model(populations[population_id])).to(
                    device
                )[None]
                latent = latent_tensor[population_id][None]
                noise = fixed_noise(population_id, target.shape[1], 10_000)
                generated = decoder(latent, noise)
                value, _ = distribution_loss(
                    generated, target, projection_tensor,
                    0.25, 0.25, 0.25, 0.20,
                )
                values.append(float(value))
        decoder.train()
        return float(np.mean(values))

    decoder_history = []
    best_decoder_validation = float("inf")
    best_decoder_step = 0
    best_decoder_state = None
    for step in range(1, args.decoder_steps + 1):
        population_id = int(rng.choice(train_ids))
        real = torch.from_numpy(to_model(populations[population_id])).to(device)[None]
        n_points = real.shape[1]
        noise = sample_decoder_noise(1, n_points, args.noise_dim, device)
        soft = decoder(latent_tensor[population_id][None], noise)
        loss, components = distribution_loss(
            soft, real, projection_tensor, 0.25, 0.25, 0.25, 0.20
        )
        decoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        decoder_optimizer.step()
        decoder_scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.decoder_steps:
            score = decoder_validation()
            row = {
                "step": step,
                "training_cells": n_points,
                "population_id": population_id,
                "batch_populations": 1,
                "training_total": float(loss.detach()),
                **{key: float(value.detach()) for key, value in components.items()},
                "validation_distribution": score,
            }
            decoder_history.append(row)
            print(json.dumps({"stage": "decoder", **row}), flush=True)
            if score < best_decoder_validation:
                best_decoder_validation = score
                best_decoder_step = step
                best_decoder_state = {
                    key: value.detach().cpu().clone()
                    for key, value in decoder.state_dict().items()
                }
            if step - best_decoder_step >= args.patience:
                break
    if best_decoder_state is None:
        raise RuntimeError("Decoder did not produce a checkpoint")
    decoder.load_state_dict(best_decoder_state)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    flow = PopulationVelocityField(
        cell_dim=cell_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
    ).to(device)
    nn.init.zeros_(flow.output.weight)
    nn.init.zeros_(flow.output.bias)
    flow_optimizer = torch.optim.AdamW(
        flow.parameters(), lr=args.flow_learning_rate, weight_decay=1e-5
    )
    flow_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        flow_optimizer, T_max=args.flow_steps
    )
    probability_path = CondOTProbPath()

    def flow_validation() -> float:
        flow.eval()
        values = []
        with torch.no_grad():
            for population_id in validation_ids:
                population_id = int(population_id)
                target = torch.from_numpy(to_model(populations[population_id])).to(
                    device
                )[None]
                latent = latent_tensor[population_id][None]
                noise = fixed_noise(population_id, target.shape[1], 20_000)
                source = decoder(latent, noise)
                corrected = integrate_corrective_flow(flow, source, args.integration_steps)
                value, _ = distribution_loss(
                    corrected, target, projection_tensor,
                    0.25, 0.25, 0.25, 0.20,
                )
                values.append(float(value))
        flow.train()
        return float(np.mean(values))

    baseline_flow_validation = flow_validation()
    best_flow_validation = baseline_flow_validation
    best_flow_step = 0
    best_flow_state = {
        key: value.detach().cpu().clone() for key, value in flow.state_dict().items()
    }
    flow_history = []
    for step in range(1, args.flow_steps + 1):
        population_id = int(rng.choice(train_ids))
        latent = latent_tensor[population_id][None]
        real = torch.from_numpy(to_model(populations[population_id])).to(device)[None]
        n_points = real.shape[1]
        with torch.no_grad():
            source = decoder(
                latent,
                sample_decoder_noise(1, n_points, args.noise_dim, device),
            )
        time, path, target_velocity = sample_flow_path(probability_path, source, real)
        predicted_velocity = flow(path.flatten(0, 1), time.flatten()).view_as(path)
        # Flow-matching objective: E[||v_theta(x_t, t) - u_t||_2^2].
        flow_matching = F.mse_loss(predicted_velocity, target_velocity)
        loss = flow_matching
        flow_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        flow_optimizer.step()
        flow_scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.flow_steps:
            score = flow_validation()
            row = {
                "step": step,
                "training_cells": n_points,
                "population_id": population_id,
                "batch_populations": 1,
                "training_total": float(loss.detach()),
                "flow_matching": float(flow_matching.detach()),
                "baseline_decoder_validation": baseline_flow_validation,
                "validation_corrected_distribution": score,
            }
            flow_history.append(row)
            print(json.dumps({"stage": "flow", **row}), flush=True)
            if score < best_flow_validation:
                best_flow_validation = score
                best_flow_step = step
                best_flow_state = {
                    key: value.detach().cpu().clone()
                    for key, value in flow.state_dict().items()
                }
            if step - best_flow_step >= args.patience:
                break
    flow.load_state_dict(best_flow_state)
    flow.eval()

    eval_rng = np.random.default_rng(args.seed + 5)
    if args.eval_populations > 0:
        selected_test = eval_rng.choice(
            test_ids, size=min(args.eval_populations, len(test_ids)), replace=False
        )
    else:
        selected_test = test_ids
    metric_projections = projection_rng.normal(size=(64, cell_dim))
    metric_projections /= np.linalg.norm(metric_projections, axis=1, keepdims=True)
    rows = []
    with torch.no_grad():
        for population_id in selected_test:
            target = populations[int(population_id)]
            n_cells = len(target)
            latent = latent_tensor[int(population_id)][None]
            source = decoder(
                latent,
                fixed_noise(int(population_id), n_cells, 30_000),
            )
            corrected = integrate_corrective_flow(flow, source, args.integration_steps)
            generated = {
                "soft_mlp_decoder": from_model(source[0].cpu().numpy()),
                "unconditional_reflow": from_model(corrected[0].cpu().numpy()),
                "real_vs_real": sample_cells(target, n_cells, eval_rng),
            }
            for method, prediction in generated.items():
                count = min(args.metric_points, len(prediction), len(target))
                prediction_sample = sample_cells(prediction, count, eval_rng)
                target_sample = sample_cells(target, count, eval_rng)
                rows.append(
                    {
                        "population_id": int(population_id),
                        "population_name": names[int(population_id)],
                        "true_count": n_cells,
                        "generated_count": len(prediction),
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
    methods = ("soft_mlp_decoder", "unconditional_reflow", "real_vs_real")
    method_summaries = {method: summarize(rows, method) for method in methods}
    summary = {
        "method": "HiPoNet-conditioned soft MLP plus unconditional flow matching",
        "flow_matching_implementation": "Meta flow_matching 1.0.10",
        "training_pairing": (
            "independent within-population pairing supplied to Meta CondOTProbPath"
        ),
        "cfm_path": "Meta flow_matching CondOTProbPath",
        "flow_objective": "pure flow-matching MSE; no auxiliary terms",
        "flow_conditioning": "x_t and t only; z conditions only the soft decoder",
        "cell_dim": cell_dim,
        "latent_dim": latent_dim,
        "decoder_input": "population latent z and per-cell Gaussian noise epsilon",
        "decoder_architecture": "one shared MLP mapping [z, epsilon_i] directly to each cell",
        "noise_dim": args.noise_dim,
        "generation_count_source": "target_population_count",
        "training_cell_count_source": "all cells in the selected training population",
        "validation_cell_count_source": "all cells in each validation population",
        "validation_populations": int(len(validation_ids)),
        "test_populations": int(len(selected_test)),
        "correlation_definition": (
            "PCC and SCC across the 44 marker means of each generated/real population"
        ),
        "best_decoder_step": best_decoder_step,
        "best_decoder_validation": best_decoder_validation,
        "best_flow_step": best_flow_step,
        "baseline_decoder_validation": baseline_flow_validation,
        "best_flow_validation": best_flow_validation,
        "flow_improved": best_flow_step > 0,
        "held_out_test_statistics": {
            method: {
                metric: {
                    "mean": method_summaries[method][metric],
                    "standard_deviation": method_summaries[method][
                        f"{metric}_standard_deviation"
                    ],
                }
                for metric in ("chamfer", "emd", "pcc", "scc")
            }
            for method in methods
        },
        "distribution_metrics": method_summaries,
    }
    torch.save(
        {
            "decoder_state_dict": best_decoder_state,
            "flow_state_dict": best_flow_state,
            "summary": summary,
            "cell_mean": cell_mean,
            "cell_std": cell_std,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
        },
        args.output_dir / "population_reflow.pt",
    )
    (args.output_dir / "decoder_history.json").write_text(
        json.dumps(decoder_history, indent=2) + "\n"
    )
    (args.output_dir / "flow_history.json").write_text(
        json.dumps(flow_history, indent=2) + "\n"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
