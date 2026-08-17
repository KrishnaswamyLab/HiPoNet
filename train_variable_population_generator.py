#!/usr/bin/env python
"""Train z-to-count, variable soft cloud, and OT corrective flow for PDO data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import nn

from models.population_flow import ConditionalPopulationFlow
from train_soft_pointcloud_corrective_flow import (
    aggregate,
    distribution_loss,
    evaluation_metrics,
    integrate_corrective_flow,
    sample_cells,
)
from train_stochastic_soft_corrective_flow import sample_soft_coupled_targets


class NegativeBinomialCountHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, initial_mean: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            self.network[-1].bias[0] = float(np.log(initial_mean))
            self.network[-1].bias[1] = 1.0

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean, raw_dispersion = self.network(latent).unbind(1)
        mean = torch.exp(log_mean.clamp(0.0, 10.0))
        dispersion = F.softplus(raw_dispersion) + 1e-4
        return mean, dispersion


def negative_binomial_nll(
    count: torch.Tensor, mean: torch.Tensor, dispersion: torch.Tensor
) -> torch.Tensor:
    log_probability = (
        torch.lgamma(count + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(count + 1.0)
        + dispersion * (torch.log(dispersion) - torch.log(dispersion + mean))
        + count * (torch.log(mean) - torch.log(dispersion + mean))
    )
    return -log_probability.mean()


class VariableSoftPointCloudDecoder(nn.Module):
    """Generate any requested number of stochastic soft cells from z."""

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
        self.query_dim = query_dim
        self.cell_dim = cell_dim
        self.maximum_scale = maximum_scale
        self.network = nn.Sequential(
            nn.Linear(latent_dim + query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * cell_dim),
        )
        initial_logit = float(np.log(np.expm1(initial_scale)))
        with torch.no_grad():
            self.network[-1].bias[cell_dim:].fill_(initial_logit)

    def forward(
        self,
        latent: torch.Tensor,
        queries: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if queries.ndim == 2:
            queries = queries[None].expand(len(latent), -1, -1)
        latent_cells = latent[:, None].expand(-1, queries.shape[1], -1)
        mean, raw_scale = self.network(
            torch.cat((latent_cells, queries), dim=2)
        ).split(self.cell_dim, dim=2)
        scale = F.softplus(raw_scale).clamp(max=self.maximum_scale)
        return mean + scale * noise, mean, scale


def sample_queries(
    batch_size: int,
    n_points: int,
    query_dim: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries = torch.randn(
        batch_size, n_points, query_dim, device=device, generator=generator
    )
    return queries


def full_population_moment_loss(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    def covariance(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(0, keepdim=True)
        return centered.T @ centered / max(len(values) - 1, 1)

    return (
        F.mse_loss(prediction.mean(0), reference.mean(0))
        + F.mse_loss(prediction.std(0, unbiased=False), reference.std(0, unbiased=False))
        + 0.1 * F.mse_loss(covariance(prediction), covariance(reference))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_cache", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_patient", default="75")
    parser.add_argument("--test_patient", default="99")
    parser.add_argument("--count_steps", type=int, default=3000)
    parser.add_argument("--decoder_steps", type=int, default=5000)
    parser.add_argument("--flow_steps", type=int, default=4000)
    parser.add_argument("--batch_populations", type=int, default=8)
    parser.add_argument("--count_batch_size", type=int, default=256)
    parser.add_argument("--training_points", type=int, default=128)
    parser.add_argument("--training_point_choices", default="")
    parser.add_argument("--full_cloud_every", type=int, default=0)
    parser.add_argument("--full_cloud_moment_weight", type=float, default=0.0)
    parser.add_argument("--metric_points", type=int, default=256)
    parser.add_argument("--query_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--initial_scale", type=float, default=0.10)
    parser.add_argument("--maximum_scale", type=float, default=1.0)
    parser.add_argument("--count_learning_rate", type=float, default=3e-4)
    parser.add_argument("--decoder_learning_rate", type=float, default=3e-4)
    parser.add_argument("--flow_learning_rate", type=float, default=5e-5)
    parser.add_argument("--ot_epsilon", type=float, default=0.10)
    parser.add_argument("--ot_iterations", type=int, default=30)
    parser.add_argument("--flow_distribution_weight", type=float, default=0.05)
    parser.add_argument("--velocity_weight", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=1600)
    parser.add_argument("--integration_steps", type=int, default=32)
    parser.add_argument("--eval_populations", type=int, default=48)
    parser.add_argument("--max_generated_cells", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1818)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_point_choices = tuple(
        int(value)
        for value in (args.training_point_choices or str(args.training_points)).split(",")
    )
    if any(value < 2 for value in training_point_choices):
        raise ValueError("Training point choices must all be at least two")
    training_cell_budget = args.batch_populations * args.training_points

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cache = np.load(args.population_cache, allow_pickle=True)
    populations = [np.asarray(value, dtype=np.float32) for value in cache["populations"]]
    names = [str(value) for value in cache["group_names"]]
    counts = np.asarray([len(value) for value in populations], dtype=np.float32)
    latents = np.asarray(np.load(args.latents), dtype=np.float32)
    if len(populations) != len(latents):
        raise ValueError("Population and latent counts do not match")
    patients = np.asarray([name.split("__", 1)[0] for name in names])
    validation_ids = np.flatnonzero(patients == str(args.validation_patient))
    test_ids = np.flatnonzero(patients == str(args.test_patient))
    train_ids = np.flatnonzero(
        (patients != str(args.validation_patient))
        & (patients != str(args.test_patient))
    )
    if min(len(train_ids), len(validation_ids), len(test_ids)) == 0:
        raise ValueError("Train, validation, and test sets must be nonempty")
    cell_dim = populations[0].shape[1]
    latent_dim = latents.shape[1]

    latent_mean = latents[train_ids].mean(0).astype(np.float32)
    latent_std = np.maximum(latents[train_ids].std(0), 1e-6).astype(np.float32)
    scaled_latents = ((latents - latent_mean) / latent_std).astype(np.float32)
    latent_tensor = torch.from_numpy(scaled_latents).to(device)
    count_tensor = torch.from_numpy(counts).to(device)

    transform_rng = np.random.default_rng(args.seed + 1)
    transform_samples = np.concatenate(
        [
            sample_cells(
                populations[int(index)],
                min(256, len(populations[int(index)])),
                transform_rng,
            )
            for index in train_ids
        ]
    )
    cell_mean = transform_samples.mean(0).astype(np.float32)
    cell_std = np.maximum(transform_samples.std(0), 1e-4).astype(np.float32)

    def to_model(values: np.ndarray) -> np.ndarray:
        return ((values - cell_mean) / cell_std).astype(np.float32)

    def from_model(values: np.ndarray) -> np.ndarray:
        return (values * cell_std + cell_mean).astype(np.float32)

    projection_rng = np.random.default_rng(args.seed + 2)
    training_projections = projection_rng.normal(size=(32, cell_dim)).astype(np.float32)
    training_projections /= np.linalg.norm(training_projections, axis=1, keepdims=True)
    training_projections_tensor = torch.from_numpy(training_projections).to(device)

    count_head = NegativeBinomialCountHead(
        latent_dim, args.hidden_dim, float(counts[train_ids].mean())
    ).to(device)
    count_optimizer = torch.optim.AdamW(
        count_head.parameters(), lr=args.count_learning_rate, weight_decay=1e-5
    )
    count_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        count_optimizer, T_max=args.count_steps
    )
    count_history = []
    best_count_state = None
    best_count_validation = float("inf")
    best_count_step = 0
    validation_index_tensor = torch.from_numpy(validation_ids).to(device)

    for step in range(1, args.count_steps + 1):
        selected = rng.choice(
            train_ids,
            size=min(args.count_batch_size, len(train_ids)),
            replace=False,
        )
        selected_tensor = torch.from_numpy(selected).to(device)
        mean, dispersion = count_head(latent_tensor[selected_tensor])
        nll = negative_binomial_nll(
            count_tensor[selected_tensor], mean, dispersion
        )
        log_count_loss = F.smooth_l1_loss(
            torch.log1p(mean), torch.log1p(count_tensor[selected_tensor])
        )
        loss = nll + 0.1 * log_count_loss
        count_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(count_head.parameters(), 5.0)
        count_optimizer.step()
        count_scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.count_steps:
            count_head.eval()
            with torch.no_grad():
                validation_mean, validation_dispersion = count_head(
                    latent_tensor[validation_index_tensor]
                )
                validation_nll = negative_binomial_nll(
                    count_tensor[validation_index_tensor],
                    validation_mean,
                    validation_dispersion,
                )
            count_head.train()
            row = {
                "step": step,
                "training_nll": float(nll.detach()),
                "training_log_count": float(log_count_loss.detach()),
                "validation_nll": float(validation_nll),
            }
            count_history.append(row)
            print(json.dumps({"stage": "count", **row}), flush=True)
            if validation_nll < best_count_validation:
                best_count_validation = float(validation_nll)
                best_count_step = step
                best_count_state = {
                    key: value.detach().cpu().clone()
                    for key, value in count_head.state_dict().items()
                }

    if best_count_state is None:
        raise RuntimeError("Count head did not produce a checkpoint")
    count_head.load_state_dict(best_count_state)
    count_head.eval()

    decoder = VariableSoftPointCloudDecoder(
        latent_dim,
        args.query_dim,
        cell_dim,
        args.hidden_dim,
        args.initial_scale,
        args.maximum_scale,
    ).to(device)
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.decoder_learning_rate, weight_decay=1e-5
    )
    decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        decoder_optimizer, T_max=args.decoder_steps
    )
    validation_rng = np.random.default_rng(args.seed + 3)
    validation_selection = validation_ids[: min(args.batch_populations, len(validation_ids))]
    validation_indices_tensor = torch.from_numpy(validation_selection).to(device)
    validation_latent = latent_tensor[validation_indices_tensor]
    validation_target = torch.from_numpy(
        np.stack(
            [
                to_model(
                    sample_cells(
                        populations[int(index)], args.training_points, validation_rng
                    )
                )
                for index in validation_selection
            ]
        )
    ).to(device)
    validation_generator = torch.Generator(device=device).manual_seed(args.seed + 4)
    validation_queries = torch.randn(
        len(validation_selection),
        args.training_points,
        args.query_dim,
        device=device,
        generator=validation_generator,
    )
    validation_noise = torch.randn(
        len(validation_selection),
        args.training_points,
        cell_dim,
        device=device,
        generator=validation_generator,
    )
    decoder_history = []
    best_decoder_state = None
    best_decoder_validation = float("inf")
    best_decoder_step = 0

    for step in range(1, args.decoder_steps + 1):
        training_points = int(rng.choice(training_point_choices))
        batch_populations = min(
            args.batch_populations,
            max(1, training_cell_budget // training_points),
        )
        selected = rng.choice(
            train_ids,
            size=batch_populations,
            replace=len(train_ids) < batch_populations,
        )
        selected_tensor = torch.from_numpy(selected).to(device)
        real = torch.from_numpy(
            np.stack(
                [
                    to_model(
                        sample_cells(populations[int(index)], training_points, rng)
                    )
                    for index in selected
                ]
            )
        ).to(device)
        queries = torch.randn(
            batch_populations,
            training_points,
            args.query_dim,
            device=device,
        )
        noise = torch.randn_like(real)
        soft, _, scale = decoder(latent_tensor[selected_tensor], queries, noise)
        loss, components = distribution_loss(
            soft,
            real,
            training_projections_tensor,
            0.25,
            0.25,
            0.25,
            0.20,
        )
        full_cloud_moment = loss.new_zeros(())
        if args.full_cloud_every > 0 and step % args.full_cloud_every == 0:
            full_index = int(rng.choice(selected))
            full_reference = torch.from_numpy(
                to_model(populations[full_index])
            ).to(device)
            full_queries = torch.randn(
                1, len(full_reference), args.query_dim, device=device
            )
            full_noise = torch.randn(
                1, len(full_reference), cell_dim, device=device
            )
            full_prediction, _, _ = decoder(
                latent_tensor[full_index][None], full_queries, full_noise
            )
            full_cloud_moment = full_population_moment_loss(
                full_prediction[0], full_reference
            )
            loss = loss + args.full_cloud_moment_weight * full_cloud_moment
        decoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        decoder_optimizer.step()
        decoder_scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.decoder_steps:
            decoder.eval()
            with torch.no_grad():
                validation_soft, _, validation_scale = decoder(
                    validation_latent, validation_queries, validation_noise
                )
                validation_loss, _ = distribution_loss(
                    validation_soft,
                    validation_target,
                    training_projections_tensor,
                    0.25,
                    0.25,
                    0.25,
                    0.20,
                )
            decoder.train()
            row = {
                "step": step,
                "training_points": training_points,
                "batch_populations": batch_populations,
                "training_total": float(loss.detach()),
                "full_cloud_moments": float(full_cloud_moment.detach()),
                "mean_scale": float(scale.detach().mean()),
                **{key: float(value.detach()) for key, value in components.items()},
                "validation_distribution": float(validation_loss),
            }
            decoder_history.append(row)
            print(json.dumps({"stage": "decoder", **row}), flush=True)
            if validation_loss < best_decoder_validation:
                best_decoder_validation = float(validation_loss)
                best_decoder_step = step
                best_decoder_state = {
                    key: value.detach().cpu().clone()
                    for key, value in decoder.state_dict().items()
                }

    if best_decoder_state is None:
        raise RuntimeError("Variable decoder did not produce a checkpoint")
    decoder.load_state_dict(best_decoder_state)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    flow = ConditionalPopulationFlow(
        cell_dim=cell_dim,
        latent_dim=latent_dim,
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
    with torch.no_grad():
        validation_source, _, _ = decoder(
            validation_latent, validation_queries, validation_noise
        )
        baseline_flow_validation, _ = distribution_loss(
            validation_source,
            validation_target,
            training_projections_tensor,
            0.25,
            0.25,
            0.25,
            0.20,
        )
    flow_history = []
    best_flow_state = {
        key: value.detach().cpu().clone() for key, value in flow.state_dict().items()
    }
    best_flow_validation = float(baseline_flow_validation)
    best_flow_step = 0

    for step in range(1, args.flow_steps + 1):
        training_points = int(rng.choice(training_point_choices))
        batch_populations = min(
            args.batch_populations,
            max(1, training_cell_budget // training_points),
        )
        selected = rng.choice(
            train_ids,
            size=batch_populations,
            replace=len(train_ids) < batch_populations,
        )
        selected_tensor = torch.from_numpy(selected).to(device)
        latent = latent_tensor[selected_tensor]
        real = torch.from_numpy(
            np.stack(
                [
                    to_model(
                        sample_cells(populations[int(index)], training_points, rng)
                    )
                    for index in selected
                ]
            )
        ).to(device)
        queries = torch.randn(
            batch_populations,
            training_points,
            args.query_dim,
            device=device,
        )
        with torch.no_grad():
            source, _, _ = decoder(latent, queries, torch.randn_like(real))
            coupled_target = sample_soft_coupled_targets(
                source, real, args.ot_epsilon, args.ot_iterations
            )
        time = torch.rand(batch_populations, training_points, 1, device=device)
        path = (1.0 - time) * source + time * coupled_target
        target_velocity = coupled_target - source
        condition = latent[:, None].expand(-1, training_points, -1)
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
            + args.flow_distribution_weight * endpoint_distribution
            + args.velocity_weight * velocity_loss
        )
        flow_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        flow_optimizer.step()
        flow_scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.flow_steps:
            flow.eval()
            with torch.no_grad():
                corrected = integrate_corrective_flow(
                    flow,
                    validation_source,
                    validation_latent,
                    args.integration_steps,
                )
                validation_loss, _ = distribution_loss(
                    corrected,
                    validation_target,
                    training_projections_tensor,
                    0.25,
                    0.25,
                    0.25,
                    0.20,
                )
            flow.train()
            row = {
                "step": step,
                "training_points": training_points,
                "batch_populations": batch_populations,
                "training_total": float(loss.detach()),
                "flow": float(flow_loss.detach()),
                "velocity": float(velocity_loss.detach()),
                **{key: float(value.detach()) for key, value in components.items()},
                "baseline_decoder_validation": float(baseline_flow_validation),
                "validation_corrected_distribution": float(validation_loss),
            }
            flow_history.append(row)
            print(json.dumps({"stage": "flow", **row}), flush=True)
            if validation_loss < best_flow_validation:
                best_flow_validation = float(validation_loss)
                best_flow_step = step
                best_flow_state = {
                    key: value.detach().cpu().clone()
                    for key, value in flow.state_dict().items()
                }
            if step - best_flow_step >= args.patience:
                break

    flow.load_state_dict(best_flow_state)
    flow.eval()

    test_indices_tensor = torch.from_numpy(test_ids).to(device)
    with torch.no_grad():
        test_count_mean, test_count_dispersion = count_head(
            latent_tensor[test_indices_tensor]
        )
    predicted_counts = test_count_mean.cpu().numpy()
    true_counts = counts[test_ids]
    constant_prediction = np.full_like(true_counts, counts[train_ids].mean())
    count_metrics = {
        "negative_binomial_nll": float(
            negative_binomial_nll(
                count_tensor[test_indices_tensor],
                test_count_mean,
                test_count_dispersion,
            )
        ),
        "mae": float(np.mean(np.abs(predicted_counts - true_counts))),
        "rmse": float(np.sqrt(np.mean((predicted_counts - true_counts) ** 2))),
        "median_absolute_error": float(np.median(np.abs(predicted_counts - true_counts))),
        "pearson": float(pearsonr(predicted_counts, true_counts).statistic),
        "spearman": float(spearmanr(predicted_counts, true_counts).statistic),
        "mean_predicted_count": float(predicted_counts.mean()),
        "mean_true_count": float(true_counts.mean()),
        "constant_mean_baseline_mae": float(
            np.mean(np.abs(constant_prediction - true_counts))
        ),
    }

    nearest_train = {}
    for population_id in test_ids:
        distances = np.linalg.norm(
            scaled_latents[train_ids] - scaled_latents[int(population_id)], axis=1
        )
        nearest_train[int(population_id)] = int(train_ids[np.argmin(distances)])
    metric_projections = projection_rng.normal(size=(64, cell_dim))
    metric_projections /= np.linalg.norm(metric_projections, axis=1, keepdims=True)
    eval_rng = np.random.default_rng(args.seed + 5)
    selected_test = eval_rng.choice(
        test_ids, size=min(args.eval_populations, len(test_ids)), replace=False
    )
    detail_rows = []
    with torch.no_grad():
        for population_id in selected_test:
            target_population = populations[int(population_id)]
            predicted_count = len(target_population)
            latent = latent_tensor[int(population_id)][None]
            queries = torch.randn(1, predicted_count, args.query_dim, device=device)
            noise = torch.randn(1, predicted_count, cell_dim, device=device)
            soft, _, _ = decoder(latent, queries, noise)
            corrected = integrate_corrective_flow(
                flow, soft, latent, args.integration_steps
            )
            generated = {
                "variable_soft_decoder": from_model(soft[0].cpu().numpy()),
                "variable_corrected_flow": from_model(corrected[0].cpu().numpy()),
                "nearest_latent_resampling": sample_cells(
                    populations[nearest_train[int(population_id)]],
                    predicted_count,
                    eval_rng,
                ),
                "real_vs_real": sample_cells(
                    populations[int(population_id)], predicted_count, eval_rng
                ),
            }
            for method, prediction in generated.items():
                metric_count = min(
                    args.metric_points, len(prediction), len(target_population)
                )
                prediction_sample = sample_cells(prediction, metric_count, eval_rng)
                target_sample = sample_cells(target_population, metric_count, eval_rng)
                detail_rows.append(
                    {
                        "population_id": int(population_id),
                        "population_name": names[int(population_id)],
                        "true_count": int(len(target_population)),
                        "predicted_count": predicted_count,
                        "method": method,
                        **evaluation_metrics(
                            prediction_sample, target_sample, metric_projections
                        ),
                    }
                )

    metrics_path = args.output_dir / "test_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    methods = (
        "variable_soft_decoder",
        "variable_corrected_flow",
        "nearest_latent_resampling",
        "real_vs_real",
    )
    distribution_metrics = {
        method: aggregate([row for row in detail_rows if row["method"] == method])
        for method in methods
    }
    summary = {
        "method": "target-count variable stochastic soft decoder plus OT-CFM",
        "generation_count_source": "target_population_count",
        "cell_dim": cell_dim,
        "latent_dim": latent_dim,
        "training_point_choices": list(training_point_choices),
        "full_cloud_every": args.full_cloud_every,
        "full_cloud_moment_weight": args.full_cloud_moment_weight,
        "best_count_step": best_count_step,
        "best_count_validation": best_count_validation,
        "best_decoder_step": best_decoder_step,
        "best_decoder_validation": best_decoder_validation,
        "best_flow_step": best_flow_step,
        "baseline_decoder_validation": float(baseline_flow_validation),
        "best_flow_validation": best_flow_validation,
        "flow_improved": best_flow_step > 0,
        "count_metrics": count_metrics,
        "distribution_metrics": distribution_metrics,
    }
    torch.save(
        {
            "count_state_dict": best_count_state,
            "decoder_state_dict": best_decoder_state,
            "flow_state_dict": best_flow_state,
            "summary": summary,
            "cell_mean": cell_mean,
            "cell_std": cell_std,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
        },
        args.output_dir / "variable_population_generator.pt",
    )
    (args.output_dir / "count_history.json").write_text(
        json.dumps(count_history, indent=2) + "\n"
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
