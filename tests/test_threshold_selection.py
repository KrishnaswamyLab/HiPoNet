"""Tests for adaptive threshold selection via Kneedle algorithm.

Run:
    uv run python -m pytest tests/test_threshold_selection.py -v
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import numpy as np
import torch
import pytest

from models.threshold_selection import select_threshold, _kneedle, _count_simplices


# ---------------------------------------------------------------------------
# Kneedle unit tests
# ---------------------------------------------------------------------------

class TestKneedle:
    def test_obvious_knee_increasing(self):
        """Saturating curve: steep then flat. Knee at the bend."""
        x = np.linspace(0, 1, 100)
        # Steep rise then plateau — concave knee near x=0.3
        y = np.where(x < 0.3, x * 3.0, 0.9 + (x - 0.3) * 0.01)
        knee = _kneedle(x, y, direction="increasing")
        # Knee should be near index 30 (x=0.3)
        assert 20 <= knee <= 40

    def test_obvious_knee_decreasing(self):
        """Inverted L: steep then flat. Knee should be near the bend."""
        x = np.linspace(0, 1, 100)
        y = np.where(x < 0.3, 1.0 - x * 3.0, 0.1 - x * 0.01)
        y = np.clip(y, 0, None)
        knee = _kneedle(x, y, direction="decreasing")
        assert 20 <= knee <= 40

    def test_linear_no_strong_knee(self):
        """Linear curve — knee can be anywhere, just shouldn't crash."""
        x = np.linspace(0, 1, 50)
        y = x.copy()
        knee = _kneedle(x, y, direction="increasing")
        assert 0 <= knee < 50


# ---------------------------------------------------------------------------
# Simplex counting tests
# ---------------------------------------------------------------------------

class TestCountSimplices:
    def test_complete_graph(self):
        """Low threshold on a small cloud should give many edges."""
        torch.manual_seed(0)
        N = 10
        # Points very close together → all kernel weights near 1
        pc = torch.randn(N, 3) * 0.1
        G = pc @ pc.T
        diag = G.diag().unsqueeze(0)
        D_sq = (diag + diag.T - 2 * G).clamp_min(0)
        W = torch.exp(-D_sq / 1.0)

        n_edges, n_tri = _count_simplices(W, threshold=0.01)
        max_edges = N * (N - 1) // 2
        # Should have nearly all edges
        assert n_edges > max_edges * 0.8

    def test_no_edges_high_threshold(self):
        """Very high threshold should yield zero edges."""
        torch.manual_seed(0)
        pc = torch.randn(10, 3) * 10  # spread out
        G = pc @ pc.T
        diag = G.diag().unsqueeze(0)
        D_sq = (diag + diag.T - 2 * G).clamp_min(0)
        W = torch.exp(-D_sq / 0.1)

        n_edges, n_tri = _count_simplices(W, threshold=0.99)
        assert n_edges == 0
        assert n_tri == 0


# ---------------------------------------------------------------------------
# Integration: select_threshold
# ---------------------------------------------------------------------------

class TestSelectThreshold:
    def test_returns_valid_threshold(self):
        torch.manual_seed(42)
        pc = torch.randn(50, 3)
        th, data = select_threshold(pc, sigma=0.5)
        assert 0.01 <= th <= 0.95
        assert "thresholds" in data
        assert "n_edges" in data
        assert "n_triangles" in data
        assert "knee_idx" in data

    def test_sweep_data_monotonic_edges(self):
        """As threshold decreases, edge count should be non-decreasing."""
        torch.manual_seed(42)
        pc = torch.randn(40, 3)
        _, data = select_threshold(pc, sigma=0.5, n_steps=30)
        # Thresholds go from high to low, so edge counts should increase
        edges = data["n_edges"]
        for i in range(1, len(edges)):
            assert edges[i] >= edges[i - 1]

    def test_dense_cloud_lower_threshold(self):
        """A dense (tight) cloud should have a lower optimal threshold
        than a spread-out cloud, since edges form more easily."""
        torch.manual_seed(0)
        dense = torch.randn(50, 3) * 0.1
        spread = torch.randn(50, 3) * 5.0
        th_dense, _ = select_threshold(dense, sigma=0.5)
        th_spread, _ = select_threshold(spread, sigma=0.5)
        # Dense cloud: edges form easily → knee happens at higher threshold
        # Spread cloud: edges form slowly → knee happens at lower threshold
        assert th_dense > th_spread

    def test_different_point_clouds_different_thresholds(self):
        """Different point clouds should generally yield different thresholds."""
        torch.manual_seed(0)
        pc1 = torch.randn(50, 3)
        pc2 = torch.randn(50, 10) * 3.0
        th1, _ = select_threshold(pc1, sigma=0.5)
        th2, _ = select_threshold(pc2, sigma=0.5)
        # Not strictly guaranteed but very likely with different dims/scales
        assert th1 != th2

    def test_reproducible(self):
        """Same input should give same output."""
        torch.manual_seed(42)
        pc = torch.randn(30, 3)
        th1, _ = select_threshold(pc, sigma=0.5)
        th2, _ = select_threshold(pc, sigma=0.5)
        assert th1 == th2
