#!/usr/bin/env python
"""Correct a pretrained soft point-cloud decoder with stochastic OT-CFM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.population_flow import ConditionalPopulationFlow
from train_soft_pointcloud_corrective_flow import (
    SoftPointCloudDecoder,
    aggregate,
    distribution_loss,
    evaluation_metrics,
    integrate_corrective_flow,
    sample_cells,
)


class ConditionalSourceScale(nn.Module):
    """Predict diagonal source uncertainty for every decoded soft cell."""

    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        cell_dim: int,
        hidden_dim: int,
        initial_scale: float,
        maximum_scale: float,
    ) -> None:
        super().__init__()
        self.initial_scale = initial_scale
        self.maximum_scale = maximum_scale
        self.network = nn.Sequential(
            nn.Linear(latent_dim + query_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cell_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        initial_logit = np.log(np.expm1(initial_scale))
        nn.init.constant_(self.network[-1].bias, float(initial_logit))

    def forward(
        self, latent: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(latent)
        latent = latent[:, None].expand(-1, len(queries), -1)
        queries = queries[None].expand(batch_size, -1, -1)
        scale = F.softplus(self.network(torch.cat((latent, queries), dim=2)))
        return scale.clamp(max=self.maximum_scale)


def sinkhorn_plan(
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Return an entropic OT plan for two equally weighted point clouds."""
    cost = torch.cdist(source, target).square()
    cost = cost / cost.detach().median().clamp_min(1e-4)
    log_kernel = -cost / epsilon
    log_mass = source.new_full((len(source),), -np.log(len(source)))
    log_u = torch.zeros_like(log_mass)
    log_v = torch.zeros_like(log_mass)
    for _ in range(iterations):
        log_u = log_mass - torch.logsumexp(log_kernel + log_v[None], dim=1)
        log_v = log_mass - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    return torch.exp(log_u[:, None] + log_kernel + log_v[None])


def sample_soft_coupled_targets(
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Sample one target for each source point from its Sinkhorn plan row."""
    coupled = []
    with torch.no_grad():
        for source_population, target_population in zip(source, target):
            plan = sinkhorn_plan(
                source_population, target_population, epsilon, iterations
            )
            probabilities = plan / plan.sum(1, keepdim=True).clamp_min(1e-12)
            target_indices = torch.multinomial(probabilities, 1).squeeze(1)
            coupled.append(target_population[target_indices])
    return torch.stack(coupled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_cache", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--soft_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_patient", default="75")
    parser.add_argument("--test_patient", default="99")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch_populations", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--query_dim", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--scale_hidden_dim", type=int, default=128)
    parser.add_argument("--initial_scale", type=float, default=0.02)
    parser.add_argument("--maximum_scale", type=float, default=0.50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--ot_epsilon", type=float, default=0.10)
    parser.add_argument("--ot_iterations", type=int, default=30)
    parser.add_argument("--distribution_weight", type=float, default=0.05)
    parser.add_argument("--velocity_weight", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=1600)
    parser.add_argument("--integration_steps", type=int, default=32)
    parser.add_argument("--eval_populations", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1817)
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
    checkpoint = torch.load(args.soft_checkpoint, map_location="cpu", weights_only=False)
    source_summary = checkpoint["summary"]
    n_points = int(source_summary["n_points"])
    cell_dim = int(source_summary["cell_dim"])
    latent_dim = int(source_summary["latent_dim"])
    if len(populations) != len(latents) or latents.shape[1] != latent_dim:
        raise ValueError("Population cache and soft-decoder latent dimensions do not match")

    patients = np.asarray([name.split("__", 1)[0] for name in names])
    validation_ids = np.flatnonzero(patients == str(args.validation_patient))
    test_ids = np.flatnonzero(patients == str(args.test_patient))
    train_ids = np.flatnonzero(
        (patients != str(args.validation_patient))
        & (patients != str(args.test_patient))
    )
    cell_mean = np.asarray(checkpoint["cell_mean"], dtype=np.float32)
    cell_std = np.asarray(checkpoint["cell_std"], dtype=np.float32)
    latent_mean = np.asarray(checkpoint["latent_mean"], dtype=np.float32)
    latent_std = np.asarray(checkpoint["latent_std"], dtype=np.float32)
    scaled_latents = ((latents - latent_mean) / latent_std).astype(np.float32)

    def to_model(values: np.ndarray) -> np.ndarray:
        return ((values - cell_mean) / cell_std).astype(np.float32)

    def from_model(values: np.ndarray) -> np.ndarray:
        return (values * cell_std + cell_mean).astype(np.float32)

    latent_tensor = torch.from_numpy(scaled_latents).to(device)
    decoder = SoftPointCloudDecoder(
        latent_dim=latent_dim,
        cell_dim=cell_dim,
        n_points=n_points,
        query_dim=args.query_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    source_scale = ConditionalSourceScale(
        latent_dim,
        args.query_dim,
        cell_dim,
        args.scale_hidden_dim,
        args.initial_scale,
        args.maximum_scale,
    ).to(device)
    flow = ConditionalPopulationFlow(
        cell_dim=cell_dim,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
    ).to(device)
    nn.init.zeros_(flow.output.weight)
    nn.init.zeros_(flow.output.bias)
    parameters = list(flow.parameters()) + list(source_scale.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    projection_rng = np.random.default_rng(args.seed + 1)
    training_projections = projection_rng.normal(size=(32, cell_dim)).astype(np.float32)
    training_projections /= np.linalg.norm(training_projections, axis=1, keepdims=True)
    training_projections_tensor = torch.from_numpy(training_projections).to(device)

    validation_rng = np.random.default_rng(args.seed + 2)
    validation_selection = validation_ids[: min(args.batch_populations, len(validation_ids))]
    validation_target = torch.from_numpy(
        np.stack(
            [
                to_model(sample_cells(populations[int(index)], n_points, validation_rng))
                for index in validation_selection
            ]
        )
    ).to(device)
    validation_indices = torch.from_numpy(validation_selection).to(device)
    validation_latent = latent_tensor[validation_indices]
    with torch.no_grad():
        validation_mean = decoder(validation_latent)
    validation_noise = torch.randn_like(validation_mean)
    with torch.no_grad():
        baseline_validation, _ = distribution_loss(
            validation_mean,
            validation_target,
            training_projections_tensor,
            0.25,
            0.25,
            0.25,
            0.20,
        )

    history = []
    best_validation = float(baseline_validation)
    best_step = 0
    best_flow_state = {
        key: value.detach().cpu().clone() for key, value in flow.state_dict().items()
    }
    best_scale_state = {
        key: value.detach().cpu().clone() for key, value in source_scale.state_dict().items()
    }

    for step in range(1, args.steps + 1):
        selected = rng.choice(
            train_ids,
            size=args.batch_populations,
            replace=len(train_ids) < args.batch_populations,
        )
        selected_indices = torch.from_numpy(selected).to(device)
        latent = latent_tensor[selected_indices]
        real = torch.from_numpy(
            np.stack(
                [
                    to_model(sample_cells(populations[int(index)], n_points, rng))
                    for index in selected
                ]
            )
        ).to(device)
        with torch.no_grad():
            source_mean = decoder(latent)
        source_std = source_scale(latent, decoder.queries.detach())
        source = source_mean + source_std * torch.randn_like(source_mean)
        coupled_target = sample_soft_coupled_targets(
            source, real, args.ot_epsilon, args.ot_iterations
        )
        time = torch.rand(args.batch_populations, n_points, 1, device=device)
        path = (1.0 - time) * source + time * coupled_target
        target_velocity = coupled_target - source
        condition = latent[:, None].expand(-1, n_points, -1)
        predicted_velocity = flow(
            path.flatten(0, 1), time.flatten(), condition.flatten(0, 1)
        ).view_as(path)
        flow_loss = F.mse_loss(predicted_velocity, target_velocity)
        endpoint = (path + (1.0 - time) * predicted_velocity).clamp(-8.0, 8.0)
        endpoint_distribution, components = distribution_loss(
            endpoint,
            real,
            training_projections_tensor,
            0.25,
            0.25,
            0.25,
            0.20,
        )
        velocity_loss = predicted_velocity.square().mean()
        loss = (
            flow_loss
            + args.distribution_weight * endpoint_distribution
            + args.velocity_weight * velocity_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            flow.eval()
            source_scale.eval()
            with torch.no_grad():
                validation_std = source_scale(
                    validation_latent, decoder.queries.detach()
                )
                validation_source = validation_mean + validation_std * validation_noise
                validation_corrected = integrate_corrective_flow(
                    flow,
                    validation_source,
                    validation_latent,
                    args.integration_steps,
                )
                validation_loss, _ = distribution_loss(
                    validation_corrected,
                    validation_target,
                    training_projections_tensor,
                    0.25,
                    0.25,
                    0.25,
                    0.20,
                )
            flow.train()
            source_scale.train()
            row = {
                "step": step,
                "training_total": float(loss.detach()),
                "flow": float(flow_loss.detach()),
                "velocity": float(velocity_loss.detach()),
                "mean_source_scale": float(source_std.detach().mean()),
                **{key: float(value.detach()) for key, value in components.items()},
                "baseline_soft_validation": float(baseline_validation),
                "validation_corrected_distribution": float(validation_loss),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if validation_loss < best_validation:
                best_validation = float(validation_loss)
                best_step = step
                best_flow_state = {
                    key: value.detach().cpu().clone()
                    for key, value in flow.state_dict().items()
                }
                best_scale_state = {
                    key: value.detach().cpu().clone()
                    for key, value in source_scale.state_dict().items()
                }
            if step - best_step >= args.patience:
                break

    flow.load_state_dict(best_flow_state)
    source_scale.load_state_dict(best_scale_state)
    flow.eval()
    source_scale.eval()

    nearest_train = {}
    for population_id in test_ids:
        distances = np.linalg.norm(
            scaled_latents[train_ids] - scaled_latents[int(population_id)], axis=1
        )
        nearest_train[int(population_id)] = int(train_ids[np.argmin(distances)])
    metric_projections = projection_rng.normal(size=(64, cell_dim))
    metric_projections /= np.linalg.norm(metric_projections, axis=1, keepdims=True)
    eval_rng = np.random.default_rng(args.seed + 3)
    selected_test = eval_rng.choice(
        test_ids, size=min(args.eval_populations, len(test_ids)), replace=False
    )
    detail_rows = []
    with torch.no_grad():
        for population_id in selected_test:
            target = sample_cells(populations[int(population_id)], n_points, eval_rng)
            latent = latent_tensor[int(population_id)][None]
            soft = decoder(latent)
            source_std = source_scale(latent, decoder.queries.detach())
            stochastic_source = soft + source_std * torch.randn_like(soft)
            corrected = integrate_corrective_flow(
                flow, stochastic_source, latent, args.integration_steps
            )
            predictions = {
                "soft_decoder": from_model(soft[0].cpu().numpy()),
                "stochastic_source": from_model(
                    stochastic_source[0].cpu().numpy()
                ),
                "corrected_flow": from_model(corrected[0].cpu().numpy()),
                "nearest_latent_resampling": sample_cells(
                    populations[nearest_train[int(population_id)]], n_points, eval_rng
                ),
                "real_vs_real": sample_cells(
                    populations[int(population_id)], n_points, eval_rng
                ),
            }
            for method, prediction in predictions.items():
                detail_rows.append(
                    {
                        "population_id": int(population_id),
                        "population_name": names[int(population_id)],
                        "method": method,
                        **evaluation_metrics(prediction, target, metric_projections),
                    }
                )

    metrics_path = args.output_dir / "test_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    methods = (
        "soft_decoder",
        "stochastic_source",
        "corrected_flow",
        "nearest_latent_resampling",
        "real_vs_real",
    )
    test_summary = {
        method: aggregate([row for row in detail_rows if row["method"] == method])
        for method in methods
    }
    summary = {
        "method": "stochastic soft source plus zero-initialized Sinkhorn OT-CFM",
        "n_points": n_points,
        "cell_dim": cell_dim,
        "latent_dim": latent_dim,
        "best_step": best_step,
        "baseline_soft_validation": float(baseline_validation),
        "best_validation": best_validation,
        "improved_validation": best_step > 0,
        "test_summary": test_summary,
    }
    torch.save(
        {
            "flow_state_dict": best_flow_state,
            "source_scale_state_dict": best_scale_state,
            "summary": summary,
        },
        args.output_dir / "stochastic_soft_corrective_flow.pt",
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
