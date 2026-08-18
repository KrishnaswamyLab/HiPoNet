import torch

from models.graph_learning import HiPoNet


def test_sparse_gwt_adapter_runs_batched_k1_forward_and_backward():
    torch.manual_seed(7)
    point_clouds = torch.randn(2, 6, 4, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, True, True, True]]
    )
    model = HiPoNet(
        dimension=4,
        n_weights=1,
        threshold=0.1,
        K=1,
        J=2,
        device="cpu",
        sigma=2.0,
        pooling=True,
        ignore_alphas=True,
    )

    features = model(point_clouds, mask)

    assert features.shape == (2, 12)
    assert torch.isfinite(features).all()
    features.sum().backward()
    assert point_clouds.grad is not None
    assert torch.isfinite(point_clouds.grad).all()


def test_sparse_gwt_adapter_retains_only_valid_nodes_without_pooling():
    point_clouds = torch.randn(2, 5, 3)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    model = HiPoNet(
        dimension=3,
        n_weights=1,
        threshold=0.1,
        K=1,
        J=2,
        device="cpu",
        sigma=2.0,
        pooling=False,
    )

    features = model(point_clouds, mask)

    assert features.shape == (int(mask.sum()), 9)
    assert torch.isfinite(features).all()
