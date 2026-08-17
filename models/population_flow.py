"""Velocity field for corrective point-cloud flow matching."""

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


class PopulationVelocityField(nn.Module):
    """Unconditional velocity field v(x_t, t) for individual cells."""

    def __init__(
        self,
        cell_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.cell_dim = cell_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.n_blocks = n_blocks
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_condition = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cell_input = nn.Linear(cell_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ConditionedResidualBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, cell_dim)

    def forward(self, cells: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 0:
            time = time.expand(len(cells))
        elif time.ndim == 2 and time.shape[1] == 1:
            time = time[:, 0]
        condition = self.time_condition(self.time_embedding(time))
        hidden = self.cell_input(cells)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(torch.nn.functional.silu(self.output_norm(hidden)))
