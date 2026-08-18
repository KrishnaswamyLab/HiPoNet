import inspect

import torch

from models.population_flow import PopulationVelocityField
from train_population_reflow import SoftPointCloudMLP, canonical_slot_batch
from utils.population_generation import integrate_corrective_flow


def test_velocity_field_uses_only_cells_and_time():
    parameters = inspect.signature(PopulationVelocityField.forward).parameters
    assert tuple(parameters) == ("self", "cells", "time")

    model = PopulationVelocityField(
        cell_dim=5, hidden_dim=16, time_dim=8, n_blocks=2
    )
    cells = torch.randn(14, 5)
    time = torch.rand(14)
    assert model(cells, time).shape == cells.shape


def test_corrective_flow_preserves_population_shape():
    model = PopulationVelocityField(
        cell_dim=5, hidden_dim=16, time_dim=8, n_blocks=2
    )
    source = torch.randn(2, 7, 5)
    corrected = integrate_corrective_flow(model, source, n_steps=3)
    assert corrected.shape == source.shape


def test_soft_decoder_is_deterministic_without_noise():
    model = SoftPointCloudMLP(
        latent_dim=4, slot_dim=6, cell_dim=5, hidden_dim=16
    )
    latent = torch.randn(2, 4)
    slots = canonical_slot_batch(2, 7, 6, torch.device("cpu"))
    first = model(latent, slots)
    second = model(latent, slots)
    assert first.shape == (2, 7, 5)
    assert torch.equal(first, second)
    assert not hasattr(model, "population_context")
