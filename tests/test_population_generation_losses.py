import torch

from utils.population_generation import soft_point_cloud_loss


def test_soft_point_cloud_loss_has_only_requested_components() -> None:
    target = torch.randn(1, 12, 5)
    prediction = target.clone().requires_grad_(True)
    loss, components = soft_point_cloud_loss(prediction, target)

    assert set(components) == {"point_cloud", "moments"}
    assert torch.isfinite(loss)
    assert float(loss.detach()) < 1e-5
    loss.backward()
    assert prediction.grad is not None


def test_soft_point_cloud_loss_respects_weights() -> None:
    prediction = torch.randn(1, 10, 4)
    target = torch.randn(1, 10, 4)
    loss, components = soft_point_cloud_loss(
        prediction, target, cloud_weight=2.0, moment_weight=3.0
    )

    expected = 2.0 * components["point_cloud"] + 3.0 * components["moments"]
    assert torch.allclose(loss, expected)
