"""Conditional flow-matching model for variable-size cell populations."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("Time embedding dimension must be even and at least four")
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = 2.0 * math.pi * torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (math.log(10_000.0) / max(half - 1, 1))
        )
        angles = time[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ConditionedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.condition = nn.Linear(condition_dim, 2 * hidden_dim)
        self.input = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=1)
        value = self.norm(hidden) * (1.0 + 0.1 * scale) + shift
        value, gate = self.input(value).chunk(2, dim=1)
        value = torch.nn.functional.silu(value) * torch.sigmoid(gate)
        return hidden + self.output(value)


class ConditionalPopulationFlow(nn.Module):
    """Velocity field v(x, t, z) conditioned on a population latent z."""

    def __init__(
        self,
        cell_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.cell_dim = cell_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.n_blocks = n_blocks
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.condition = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
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
        self,
        cells: torch.Tensor,
        time: torch.Tensor,
        population_latent: torch.Tensor,
    ) -> torch.Tensor:
        if time.ndim == 0:
            time = time.expand(len(cells))
        elif time.ndim == 2 and time.shape[1] == 1:
            time = time[:, 0]
        condition = self.condition(
            torch.cat((population_latent, self.time_embedding(time)), dim=1)
        )
        hidden = self.cell_input(cells)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(torch.nn.functional.silu(self.output_norm(hidden)))


class CanonicalSetPopulationFlow(nn.Module):
    """Set-aware velocity field conditioned on latent, time, and canonical slots."""

    def __init__(
        self,
        cell_dim: int,
        latent_dim: int,
        slot_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.global_condition = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_condition = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.cell_input = nn.Linear(cell_dim, hidden_dim)
        self.context = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_blocks)
            ]
        )
        self.blocks = nn.ModuleList(
            [ConditionedResidualBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, cell_dim)

    def forward(
        self,
        cells: torch.Tensor,
        time: torch.Tensor,
        population_latent: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        if cells.ndim != 3:
            raise ValueError("cells must have shape [batch, points, features]")
        if time.ndim == 0:
            time = time.expand(len(cells))
        elif time.ndim > 1:
            time = time.reshape(len(cells), -1)[:, 0]
        global_condition = self.global_condition(
            torch.cat((population_latent, self.time_embedding(time)), dim=1)
        )
        slot_condition = self.slot_condition(slots)
        hidden = self.cell_input(cells)
        for context_layer, block in zip(self.context, self.blocks):
            population_context = context_layer(hidden.mean(1))
            condition = (
                global_condition[:, None]
                + slot_condition
                + population_context[:, None]
            )
            hidden = block(
                hidden.flatten(0, 1), condition.flatten(0, 1)
            ).view_as(hidden)
        return self.output(torch.nn.functional.silu(self.output_norm(hidden)))


def sample_population(
    model: ConditionalPopulationFlow,
    population_latent: torch.Tensor,
    n_cells: int,
    n_steps: int = 50,
    method: str = "heun",
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the learned velocity field from Gaussian noise to cells."""
    if n_cells < 1 or n_steps < 1:
        raise ValueError("n_cells and n_steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'")
    if population_latent.ndim == 1:
        population_latent = population_latent.unsqueeze(0)
    if len(population_latent) == 1:
        population_latent = population_latent.expand(n_cells, -1)
    elif len(population_latent) != n_cells:
        raise ValueError("Population latent must have one or n_cells rows")

    cells = noise
    if cells is None:
        cells = torch.randn(
            n_cells,
            model.cell_dim,
            device=population_latent.device,
            dtype=population_latent.dtype,
        )
    step_size = 1.0 / n_steps
    for step in range(n_steps):
        time_value = step / n_steps
        time = torch.full(
            (n_cells,), time_value, device=cells.device, dtype=cells.dtype
        )
        velocity = model(cells, time, population_latent)
        if method == "euler":
            cells = cells + step_size * velocity
            continue
        proposal = cells + step_size * velocity
        next_time = torch.full_like(time, (step + 1) / n_steps)
        next_velocity = model(proposal, next_time, population_latent)
        cells = cells + 0.5 * step_size * (velocity + next_velocity)
    return cells
