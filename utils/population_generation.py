"""Losses, integration, sampling, and metrics for population generation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

from models.population_flow import PopulationVelocityField


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


def soft_point_cloud_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    cloud_weight: float = 1.0,
    moment_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sinkhorn cloud fidelity plus population moments for the soft decoder."""
    cloud_values, moment_values = [], []
    for generated, real in zip(prediction, reference):
        cloud_values.append(sinkhorn_divergence(generated, real))
        moment_values.append(full_population_moment_loss(generated, real))
    components = {
        "point_cloud": torch.stack(cloud_values).mean(),
        "moments": torch.stack(moment_values).mean(),
    }
    total = (
        cloud_weight * components["point_cloud"]
        + moment_weight * components["moments"]
    )
    return total, components


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


def full_population_moment_loss(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    return (
        F.mse_loss(prediction.mean(0), reference.mean(0))
        + F.mse_loss(prediction.std(0, unbiased=False), reference.std(0, unbiased=False))
        + 0.1 * F.mse_loss(covariance(prediction), covariance(reference))
    )


def sample_cells(
    population: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    indices = rng.choice(len(population), size=count, replace=len(population) < count)
    return population[indices].astype(np.float32)


def integrate_corrective_flow(
    model: PopulationVelocityField,
    source: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    cells = source
    step_size = 1.0 / n_steps
    for step in range(n_steps):
        time = cells.new_full(cells.shape[:2], step / n_steps)
        velocity = model(cells.flatten(0, 1), time.flatten()).view_as(cells)
        proposal = cells + step_size * velocity
        next_time = cells.new_full(cells.shape[:2], (step + 1) / n_steps)
        next_velocity = model(proposal.flatten(0, 1), next_time.flatten()).view_as(cells)
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
                    (np.cov(prediction, rowvar=False) - np.cov(target, rowvar=False))
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
