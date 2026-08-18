import torch

from train_direct_population_flow import (
    LatentConditionedVelocityField,
    fixed_gaussian,
    integrate_direct_flow,
)


def test_direct_velocity_field_shapes() -> None:
    model = LatentConditionedVelocityField(
        cell_dim=5, latent_dim=3, hidden_dim=16, time_dim=8, n_blocks=2
    )
    cells = torch.randn(7, 5)
    time = torch.rand(7)
    latent = torch.randn(7, 3)
    assert model(cells, time, latent).shape == cells.shape


def test_zero_velocity_preserves_gaussian_source() -> None:
    model = LatentConditionedVelocityField(
        cell_dim=5, latent_dim=3, hidden_dim=16, time_dim=8, n_blocks=2
    )
    torch.nn.init.zeros_(model.output.weight)
    torch.nn.init.zeros_(model.output.bias)
    source = torch.randn(2, 7, 5)
    latent = torch.randn(2, 3)
    result = integrate_direct_flow(model, source, latent, n_steps=4)
    assert torch.equal(result, source)


def test_fixed_gaussian_is_reproducible() -> None:
    device = torch.device("cpu")
    first = fixed_gaussian(9, 11, 5, 100, device)
    second = fixed_gaussian(9, 11, 5, 100, device)
    assert torch.equal(first, second)
