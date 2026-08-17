import torch

from models.graph_learning import compute_diffusion_from_dist


def test_compute_diffusion_from_dist_normalizes_across_last_axis_for_batched_graphs():
    W = torch.tensor([[[[1.0, 0.2], [0.2, 1.0]]]], dtype=torch.float32)
    mask = torch.ones((1, 2), dtype=torch.bool)

    out = compute_diffusion_from_dist(W, sigma=1.0, threshold=0.0, mask=mask)

    row_sums = out[0, 0].sum(dim=-1)
    assert torch.allclose(row_sums, torch.tensor([1.5, 1.5], dtype=torch.float32))
