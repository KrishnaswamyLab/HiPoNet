#!/usr/bin/env python
"""Decode HiPoNet latents to soft point clouds, then correct them with flow matching."""

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
from scipy.stats import pearsonr, spearmanr
from torch import nn

from models.population_flow import ConditionalPopulationFlow, sample_population


class SoftPointCloudDecoder(nn.Module):
    """Map one population latent to a fixed-size unordered set of soft cells."""

    def __init__(
        self,
        latent_dim: int,
        cell_dim: int,
        n_points: int,
        query_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.n_points = n_points
        self.queries = nn.Parameter(torch.randn(n_points, query_dim) * 0.02)
        self.network = nn.Sequential(
            nn.Linear(latent_dim + query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cell_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = len(latent)
        latent = latent[:, None, :].expand(-1, self.n_points, -1)
        queries = self.queries[None].expand(batch_size, -1, -1)
        return self.network(torch.cat((latent, queries), dim=2))


def covariance(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(0, keepdim=True)
    return centered.T @ centered / max(len(values) - 1, 1)


def population_losses(
    prediction: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    moment = (
        F.mse_loss(prediction.mean(0), reference.mean(0))
        + F.mse_loss(prediction.std(0), reference.std(0))
        + 0.1 * F.mse_loss(covariance(prediction), covariance(reference))
    )
    prediction_distances = torch.pdist(prediction)
    reference_distances = torch.pdist(reference)
    diversity = F.mse_loss(
        prediction_distances.mean(), reference_distances.mean()
    ) + F.mse_loss(prediction_distances.std(), reference_distances.std())
    return moment, diversity


def sinkhorn_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.2,
    iterations: int = 20,
) -> torch.Tensor:
    cost = torch.cdist(x, y).square()
    cost = cost / cost.detach().median().clamp_min(1e-4)
    log_kernel = -cost / epsilon
    log_a = x.new_full((len(x),), -np.log(len(x)))
    log_b = y.new_full((len(y),), -np.log(len(y)))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    plan = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
    return (plan * cost).sum()


def sinkhorn_divergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        sinkhorn_cost(x, y)
        - 0.5 * sinkhorn_cost(x, x)
        - 0.5 * sinkhorn_cost(y, y)
    ).clamp_min(0.0)


def sliced_wasserstein_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    projections: torch.Tensor,
) -> torch.Tensor:
    prediction_projected = torch.sort(prediction @ projections.T, dim=0).values
    reference_projected = torch.sort(reference @ projections.T, dim=0).values
    return F.mse_loss(prediction_projected, reference_projected)


def distribution_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    projections: torch.Tensor,
    sinkhorn_weight: float,
    swd_weight: float,
    moment_weight: float,
    diversity_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sinkhorn_values, swd_values, moment_values, diversity_values = [], [], [], []
    for generated, real in zip(prediction, reference):
        sinkhorn_values.append(sinkhorn_divergence(generated, real))
        swd_values.append(sliced_wasserstein_loss(generated, real, projections))
        moment, diversity = population_losses(generated, real)
        moment_values.append(moment)
        diversity_values.append(diversity)
    components = {
        "sinkhorn": torch.stack(sinkhorn_values).mean(),
        "swd": torch.stack(swd_values).mean(),
        "moments": torch.stack(moment_values).mean(),
        "diversity": torch.stack(diversity_values).mean(),
    }
    total = (
        sinkhorn_weight * components["sinkhorn"]
        + swd_weight * components["swd"]
        + moment_weight * components["moments"]
        + diversity_weight * components["diversity"]
    )
    return total, components


def sample_cells(
    population: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    indices = rng.choice(len(population), size=count, replace=len(population) < count)
    return population[indices].astype(np.float32)


def ot_align(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align target cells to source queries using a minimum-cost assignment."""
    aligned = []
    for source_population, target_population in zip(source, target):
        cost = torch.cdist(source_population.detach(), target_population.detach()).square()
        _, target_indices = linear_sum_assignment(cost.cpu().numpy())
        aligned.append(target_population[torch.as_tensor(target_indices, device=target.device)])
    return torch.stack(aligned)


def integrate_corrective_flow(
    model: ConditionalPopulationFlow,
    source: torch.Tensor,
    latent: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    cells = source
    condition = latent[:, None, :].expand(-1, cells.shape[1], -1)
    step_size = 1.0 / n_steps
    for step in range(n_steps):
        time = cells.new_full(cells.shape[:2], step / n_steps)
        velocity = model(
            cells.flatten(0, 1), time.flatten(), condition.flatten(0, 1)
        ).view_as(cells)
        proposal = cells + step_size * velocity
        next_time = cells.new_full(cells.shape[:2], (step + 1) / n_steps)
        next_velocity = model(
            proposal.flatten(0, 1), next_time.flatten(), condition.flatten(0, 1)
        ).view_as(cells)
        cells = cells + 0.5 * step_size * (velocity + next_velocity)
    return cells


def chamfer(prediction: np.ndarray, target: np.ndarray) -> float:
    distances = cdist(prediction, target, metric="sqeuclidean")
    return float(distances.min(1).mean() + distances.min(0).mean())


def sliced_wasserstein_metric(
    prediction: np.ndarray, target: np.ndarray, projections: np.ndarray
) -> float:
    prediction_projected = np.sort(prediction @ projections.T, axis=0)
    target_projected = np.sort(target @ projections.T, axis=0)
    return float(np.sqrt(np.mean((prediction_projected - target_projected) ** 2)))


def evaluation_metrics(
    prediction: np.ndarray, target: np.ndarray, projections: np.ndarray
) -> dict[str, float]:
    prediction_diversity = float(np.mean(cdist(prediction, prediction)))
    target_diversity = float(np.mean(cdist(target, target)))
    prediction_profile = prediction.mean(0)
    target_profile = target.mean(0)
    if prediction_profile.std() == 0 or target_profile.std() == 0:
        pcc = scc = 0.0
    else:
        pcc = float(pearsonr(prediction_profile, target_profile).statistic)
        scc = float(spearmanr(prediction_profile, target_profile).statistic)
        pcc = pcc if np.isfinite(pcc) else 0.0
        scc = scc if np.isfinite(scc) else 0.0
    return {
        "chamfer": chamfer(prediction, target),
        "pcc": pcc,
        "scc": scc,
        "sliced_wasserstein": sliced_wasserstein_metric(
            prediction, target, projections
        ),
        "marker_mean_rmse": float(
            np.sqrt(np.mean((prediction.mean(0) - target.mean(0)) ** 2))
        ),
        "marker_std_rmse": float(
            np.sqrt(np.mean((prediction.std(0) - target.std(0)) ** 2))
        ),
        "covariance_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        np.cov(prediction, rowvar=False)
                        - np.cov(target, rowvar=False)
                    )
                    ** 2
                )
            )
        ),
        "generated_diversity": prediction_diversity,
        "target_diversity": target_diversity,
        "diversity_ratio": prediction_diversity / max(target_diversity, 1e-12),
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    keys = [
        "chamfer",
        "pcc",
        "scc",
        "sliced_wasserstein",
        "marker_mean_rmse",
        "marker_std_rmse",
        "covariance_rmse",
        "generated_diversity",
        "target_diversity",
        "diversity_ratio",
    ]
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = float(values.mean())
        result[f"{key}_standard_deviation"] = float(
            values.std(ddof=1) if len(values) > 1 else 0.0
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_cache", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_patient", default="75")
    parser.add_argument("--test_patient", default="99")
    parser.add_argument("--n_points", type=int, default=128)
    parser.add_argument("--decoder_steps", type=int, default=4000)
    parser.add_argument("--flow_steps", type=int, default=6000)
    parser.add_argument("--batch_populations", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--query_dim", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--sinkhorn_weight", type=float, default=0.25)
    parser.add_argument("--swd_weight", type=float, default=0.25)
    parser.add_argument("--moment_weight", type=float, default=0.25)
    parser.add_argument("--diversity_weight", type=float, default=0.20)
    parser.add_argument("--flow_distribution_weight", type=float, default=0.10)
    parser.add_argument("--source_jitter", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=1600)
    parser.add_argument("--integration_steps", type=int, default=32)
    parser.add_argument("--eval_populations", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1816)
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
        raise ValueError("Train, validation, and test splits must all be nonempty")
    cell_dim = populations[0].shape[1]

    latent_mean = latents[train_ids].mean(0).astype(np.float32)
    latent_std = np.maximum(latents[train_ids].std(0), 1e-6).astype(np.float32)
    scaled_latents = ((latents - latent_mean) / latent_std).astype(np.float32)
    transform_rng = np.random.default_rng(args.seed + 1)
    transform_samples = np.concatenate(
        [
            sample_cells(
                populations[int(population_id)],
                min(256, len(populations[int(population_id)])),
                transform_rng,
            )
            for population_id in train_ids
        ]
    )
    cell_mean = transform_samples.mean(0).astype(np.float32)
    cell_std = np.maximum(transform_samples.std(0), 1e-4).astype(np.float32)

    def to_model(values: np.ndarray) -> np.ndarray:
        return ((values - cell_mean) / cell_std).astype(np.float32)

    def from_model(values: np.ndarray) -> np.ndarray:
        return (values * cell_std + cell_mean).astype(np.float32)

    latent_tensor = torch.from_numpy(scaled_latents).to(device)
    projection_rng = np.random.default_rng(args.seed + 2)
    training_projections = projection_rng.normal(size=(32, cell_dim)).astype(np.float32)
    training_projections /= np.linalg.norm(training_projections, axis=1, keepdims=True)
    training_projections_tensor = torch.from_numpy(training_projections).to(device)

    validation_rng = np.random.default_rng(args.seed + 3)
    validation_selection = validation_ids[: min(args.batch_populations, len(validation_ids))]
    validation_target = torch.from_numpy(
        np.stack(
            [
                to_model(
                    sample_cells(
                        populations[int(population_id)], args.n_points, validation_rng
                    )
                )
                for population_id in validation_selection
            ]
        )
    ).to(device)
    validation_latent = latent_tensor[torch.from_numpy(validation_selection).to(device)]

    decoder = SoftPointCloudDecoder(
        latent_dim=latents.shape[1],
        cell_dim=cell_dim,
        n_points=args.n_points,
        query_dim=args.query_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        decoder_optimizer, T_max=args.decoder_steps
    )
    decoder_history = []
    best_decoder_state = None
    best_decoder_validation = float("inf")
    best_decoder_step = 0

    for step in range(1, args.decoder_steps + 1):
        selected = rng.choice(
            train_ids,
            size=args.batch_populations,
            replace=len(train_ids) < args.batch_populations,
        )
        real = torch.from_numpy(
            np.stack(
                [
                    to_model(
                        sample_cells(populations[int(index)], args.n_points, rng)
                    )
                    for index in selected
                ]
            )
        ).to(device)
        selected_tensor = torch.from_numpy(selected).to(device)
        soft = decoder(latent_tensor[selected_tensor])
        loss, components = distribution_loss(
            soft,
            real,
            training_projections_tensor,
            args.sinkhorn_weight,
            args.swd_weight,
            args.moment_weight,
            args.diversity_weight,
        )
        decoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        decoder_optimizer.step()
        decoder_scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.decoder_steps:
            decoder.eval()
            with torch.no_grad():
                validation_soft = decoder(validation_latent)
                validation_loss, _ = distribution_loss(
                    validation_soft,
                    validation_target,
                    training_projections_tensor,
                    args.sinkhorn_weight,
                    args.swd_weight,
                    args.moment_weight,
                    args.diversity_weight,
                )
            decoder.train()
            row = {
                "step": step,
                "training_total": float(loss.detach()),
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
        raise RuntimeError("Soft decoder did not produce a checkpoint")
    decoder.load_state_dict(best_decoder_state)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    flow = ConditionalPopulationFlow(
        cell_dim=cell_dim,
        latent_dim=latents.shape[1],
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
    ).to(device)
    flow_optimizer = torch.optim.AdamW(
        flow.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    flow_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        flow_optimizer, T_max=args.flow_steps
    )
    flow_history = []
    best_flow_state = None
    best_flow_validation = float("inf")
    best_flow_step = 0

    for step in range(1, args.flow_steps + 1):
        selected = rng.choice(
            train_ids,
            size=args.batch_populations,
            replace=len(train_ids) < args.batch_populations,
        )
        selected_tensor = torch.from_numpy(selected).to(device)
        latent = latent_tensor[selected_tensor]
        real = torch.from_numpy(
            np.stack(
                [
                    to_model(
                        sample_cells(populations[int(index)], args.n_points, rng)
                    )
                    for index in selected
                ]
            )
        ).to(device)
        with torch.no_grad():
            source = decoder(latent)
            source = source + args.source_jitter * torch.randn_like(source)
            aligned_target = ot_align(source, real)
        time = torch.rand(args.batch_populations, args.n_points, 1, device=device)
        path = (1.0 - time) * source + time * aligned_target
        target_velocity = aligned_target - source
        condition = latent[:, None, :].expand(-1, args.n_points, -1)
        predicted_velocity = flow(
            path.flatten(0, 1), time.flatten(), condition.flatten(0, 1)
        ).view_as(path)
        flow_loss = F.mse_loss(predicted_velocity, target_velocity)
        endpoint = (path + (1.0 - time) * predicted_velocity).clamp(-8.0, 8.0)
        endpoint_distribution, components = distribution_loss(
            endpoint,
            real,
            training_projections_tensor,
            args.sinkhorn_weight,
            args.swd_weight,
            args.moment_weight,
            args.diversity_weight,
        )
        loss = flow_loss + args.flow_distribution_weight * endpoint_distribution
        flow_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        flow_optimizer.step()
        flow_scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.flow_steps:
            flow.eval()
            with torch.no_grad():
                validation_source = decoder(validation_latent)
                corrected = integrate_corrective_flow(
                    flow,
                    validation_source,
                    validation_latent,
                    min(args.integration_steps, 16),
                )
                validation_loss, _ = distribution_loss(
                    corrected,
                    validation_target,
                    training_projections_tensor,
                    args.sinkhorn_weight,
                    args.swd_weight,
                    args.moment_weight,
                    args.diversity_weight,
                )
            flow.train()
            row = {
                "step": step,
                "training_total": float(loss.detach()),
                "flow": float(flow_loss.detach()),
                **{key: float(value.detach()) for key, value in components.items()},
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

    if best_flow_state is None:
        raise RuntimeError("Corrective flow did not produce a checkpoint")
    flow.load_state_dict(best_flow_state)
    flow.eval()

    nearest_train = {}
    for population_id in test_ids:
        distances = np.linalg.norm(
            scaled_latents[train_ids] - scaled_latents[int(population_id)], axis=1
        )
        nearest_train[int(population_id)] = int(train_ids[np.argmin(distances)])
    metric_projections = projection_rng.normal(size=(64, cell_dim))
    metric_projections /= np.linalg.norm(metric_projections, axis=1, keepdims=True)
    eval_rng = np.random.default_rng(args.seed + 4)
    selected_test = eval_rng.choice(
        test_ids, size=min(args.eval_populations, len(test_ids)), replace=False
    )
    detail_rows = []
    with torch.no_grad():
        for population_id in selected_test:
            target = sample_cells(
                populations[int(population_id)], args.n_points, eval_rng
            )
            target_model = to_model(target)
            latent = latent_tensor[int(population_id)][None]
            soft_model = decoder(latent)
            corrected_model = integrate_corrective_flow(
                flow, soft_model, latent, args.integration_steps
            )
            predictions = {
                "soft_decoder": from_model(soft_model[0].cpu().numpy()),
                "corrected_flow": from_model(corrected_model[0].cpu().numpy()),
                "nearest_latent_resampling": sample_cells(
                    populations[nearest_train[int(population_id)]],
                    args.n_points,
                    eval_rng,
                ),
                "real_vs_real": sample_cells(
                    populations[int(population_id)], args.n_points, eval_rng
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
        "corrected_flow",
        "nearest_latent_resampling",
        "real_vs_real",
    )
    test_summary = {
        method: aggregate([row for row in detail_rows if row["method"] == method])
        for method in methods
    }
    summary = {
        "method": "HiPoNet soft point cloud plus OT corrective flow",
        "cell_dim": cell_dim,
        "latent_dim": int(latents.shape[1]),
        "n_points": args.n_points,
        "n_train": int(len(train_ids)),
        "n_validation": int(len(validation_ids)),
        "n_test": int(len(test_ids)),
        "best_decoder_step": best_decoder_step,
        "best_decoder_validation": best_decoder_validation,
        "best_flow_step": best_flow_step,
        "best_flow_validation": best_flow_validation,
        "source_jitter": args.source_jitter,
        "test_summary": test_summary,
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
        args.output_dir / "soft_decoder_corrective_flow.pt",
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
