"""Tests for DeepSet pooling attention in HiPoNet.

Covers:
  - PoolingAttention: output shape, gradient flow, weight storage
  - SimplicialFeatLearningLayer: with/without attention produces same shape
  - HiPoNet.get_attention_weights: returns correct structure

Run:
    uv run python -m pytest tests/test_attention.py -v
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import torch
import pytest

from models.SWT import PoolingAttention, SimplicialWaveletTransform
from models.graph_learning import SimplicialFeatLearningLayer, HiPoNet


# ---------------------------------------------------------------------------
# PoolingAttention unit tests
# ---------------------------------------------------------------------------

class TestPoolingAttention:
    def test_output_shape(self):
        feat_dim = 10
        n_elements = 20
        attn = PoolingAttention(feat_dim)
        x = torch.randn(n_elements, feat_dim)
        out = attn(x)
        assert out.shape == (feat_dim,)

    def test_weights_stored(self):
        feat_dim = 8
        attn = PoolingAttention(feat_dim)
        x = torch.randn(15, feat_dim)
        attn(x)
        assert attn.last_weights is not None
        assert attn.last_weights.shape == (15,)

    def test_weights_sum_to_one(self):
        attn = PoolingAttention(5)
        x = torch.randn(10, 5)
        attn(x)
        assert torch.allclose(attn.last_weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_gradient_flows(self):
        feat_dim = 6
        attn = PoolingAttention(feat_dim)
        x = torch.randn(8, feat_dim, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        # Gradients should flow to input
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        # Gradients should flow to attention parameters
        for p in attn.parameters():
            assert p.grad is not None

    def test_single_element(self):
        """With a single element, attention weight should be 1.0."""
        attn = PoolingAttention(4)
        x = torch.randn(1, 4)
        out = attn(x)
        assert out.shape == (4,)
        assert torch.allclose(attn.last_weights, torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# Integration with SimplicialFeatLearningLayer
# ---------------------------------------------------------------------------

class TestSimplicialAttention:
    @pytest.fixture
    def small_point_cloud(self):
        """A small batch of point clouds for testing."""
        torch.manual_seed(42)
        B, N, d = 2, 30, 5
        pc = torch.randn(B, N, d)
        mask = torch.ones(B, N, dtype=torch.bool)
        return pc, mask, d

    def test_attention_same_output_shape(self, small_point_cloud):
        """Output shape should be the same with and without attention."""
        pc, mask, d = small_point_cloud
        device = "cpu"

        layer_no_attn = SimplicialFeatLearningLayer(
            n_weights=1, dimension=d, threshold=0.3, sigma=0.5,
            J=2, device=device, max_simplex_dim=2, use_attention=False,
        )
        layer_attn = SimplicialFeatLearningLayer(
            n_weights=1, dimension=d, threshold=0.3, sigma=0.5,
            J=2, device=device, max_simplex_dim=2, use_attention=True,
        )
        # Copy alphas so both layers build the same complex
        layer_attn.alphas.data.copy_(layer_no_attn.alphas.data)

        out_no_attn = layer_no_attn(pc, mask)
        out_attn = layer_attn(pc, mask)

        assert out_no_attn.shape == out_attn.shape

    def test_attention_weights_accessible(self, small_point_cloud):
        """After forward pass, attention weights should be retrievable."""
        pc, mask, d = small_point_cloud
        layer = SimplicialFeatLearningLayer(
            n_weights=1, dimension=d, threshold=0.3, sigma=0.5,
            J=2, device="cpu", max_simplex_dim=2, use_attention=True,
        )
        layer(pc, mask)
        weights = layer.get_attention_weights()
        assert len(weights) == 3  # vertices, edges, triangles
        for k, w in weights.items():
            assert w is not None
            # Some simplex dimensions may have no simplices (e.g., no triangles)
            if w.numel() > 0:
                assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)

    def test_no_attention_returns_empty(self, small_point_cloud):
        pc, mask, d = small_point_cloud
        layer = SimplicialFeatLearningLayer(
            n_weights=1, dimension=d, threshold=0.3, sigma=0.5,
            J=2, device="cpu", max_simplex_dim=2, use_attention=False,
        )
        layer(pc, mask)
        assert layer.get_attention_weights() == {}

    def test_gradient_through_attention(self, small_point_cloud):
        """Gradients should flow through attention to alphas."""
        pc, mask, d = small_point_cloud
        layer = SimplicialFeatLearningLayer(
            n_weights=1, dimension=d, threshold=0.3, sigma=0.5,
            J=2, device="cpu", max_simplex_dim=2, use_attention=True,
        )
        out = layer(pc, mask)
        loss = out.sum()
        loss.backward()
        # Attention parameters should have gradients
        for p in layer.pool_attn.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# HiPoNet integration
# ---------------------------------------------------------------------------

class TestHiPoNetAttention:
    def test_get_attention_weights_k2(self):
        torch.manual_seed(0)
        B, N, d = 2, 25, 4
        pc = torch.randn(B, N, d)
        mask = torch.ones(B, N, dtype=torch.bool)

        model = HiPoNet(
            dimension=d, n_weights=1, threshold=0.3, K=2, J=2,
            device="cpu", sigma=0.5, use_attention=True,
        )
        model(pc, mask)
        weights = model.get_attention_weights()
        assert len(weights) == 3

    def test_get_attention_weights_k1_empty(self):
        """K=1 path should return empty attention weights."""
        model = HiPoNet(
            dimension=4, n_weights=1, threshold=0.3, K=1, J=2,
            device="cpu", sigma=0.5, use_attention=True,
        )
        assert model.get_attention_weights() == {}
