"""Losses, integration, sampling, and metrics for population generation."""

from __future__ import annotations

import numpy as np
import ot
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

from models.population_flow import PopulationVelocityField

def emd_point_cloud_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    solver: str = "sinkhorn_log",
    regularization: float = 0.2,
    iterations: int = 100,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable POT EMD for the soft decoder."""
    if solver not in {"exact", "sinkhorn_log"}:
        raise ValueError(f"Unknown POT EMD solver: {solver}")
    emd_values = []
    for generated, real in zip(prediction, reference):
        generated_weights = ot.unif(len(generated), type_as=generated)
        real_weights = ot.unif(len(real), type_as=real)
        cost = ot.dist(generated, real, metric="euclidean")
        if solver == "exact":
            value = ot.emd2(generated_weights, real_weights, cost)
        else:
            value = ot.sinkhorn2(
                generated_weights,
                real_weights,
                cost,
                reg=regularization,
                method="sinkhorn_log",
                numItermax=iterations,
                stopThr=1e-6,
                warn=False,
            )
        emd_values.append(value)
    emd = torch.stack(emd_values).mean()
    return emd, {"emd": emd}


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
