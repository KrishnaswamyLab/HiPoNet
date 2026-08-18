import torch

from utils.population_generation import emd_point_cloud_loss


def test_emd_point_cloud_loss_is_the_only_decoder_component() -> None:
    target = torch.randn(1, 12, 5)
    prediction = target.clone().requires_grad_(True)
    loss, components = emd_point_cloud_loss(prediction, target, solver="exact")

    assert set(components) == {"emd"}
    assert torch.equal(loss, components["emd"])
    assert torch.isfinite(loss)
    assert float(loss.detach()) < 1e-3
    loss.backward()
    assert prediction.grad is not None


def test_pot_sinkhorn_emd_supports_gradients_and_unequal_clouds() -> None:
    prediction = torch.randn(1, 7, 5, requires_grad=True)
    target = torch.randn(1, 9, 5)

    loss, components = emd_point_cloud_loss(
        prediction,
        target,
        solver="sinkhorn_log",
        regularization=0.2,
        iterations=30,
    )

    assert torch.equal(loss, components["emd"])
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
