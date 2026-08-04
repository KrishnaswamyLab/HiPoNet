"""Tests for diffusion-distance-based geometric Hodge Laplacian pipeline.

Covers:
  - cayley_menger_volumes: correctness for edges, triangles; differentiability
  - geometric_hodge_laplacian_torch: combinatorial recovery, PSD, shape
  - Diffusion distance computation (symmetric, non-negative)
  - SWT._build_geometric_transition_matrices: P_U/P_L set, shapes, wavelet coeff

Run:
    uv run python -m pytest tests/test_diffusion_laplacian.py -v
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import numpy as np
import torch
import pytest
from scipy.linalg import eigh

from models.hodge_laplacian import cayley_menger_volumes, geometric_hodge_laplacian_torch
from models.simplicial_complex import SimplicialComplex
from models.SWT import SimplicialWaveletTransform
from models.graph_learning import SimplicialFeatLearningLayer, compute_dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sq_dist(points):
    """Pairwise squared Euclidean distances as torch tensor."""
    pts = torch.tensor(points, dtype=torch.float32)
    diff = pts.unsqueeze(0) - pts.unsqueeze(1)
    return (diff ** 2).sum(-1)


def random_adj(n_pts, threshold=0.5, seed=0):
    """Simple Gaussian adjacency matrix for testing SWT."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_pts, 3))
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(pts))
    W = np.exp(-D ** 2 / 0.5)
    W[D > 2.0] = 0.0
    np.fill_diagonal(W, 0)
    return torch.tensor(W, dtype=torch.float32), pts


# ---------------------------------------------------------------------------
# cayley_menger_volumes — edges
# ---------------------------------------------------------------------------

class TestCayleyMengerEdges:
    """For a 1-simplex (edge), vol = distance between vertices."""

    def test_unit_edge(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        sq = make_sq_dist(pts)
        edges = torch.tensor([[0, 1]])
        vol = cayley_menger_volumes(edges, sq)
        assert vol.shape == (1,)
        assert abs(vol.item() - 1.0) < 1e-5

    def test_multiple_edges(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
        sq = make_sq_dist(pts)
        edges = torch.tensor([[0, 1], [0, 2]])
        vols = cayley_menger_volumes(edges, sq)
        assert vols.shape == (2,)
        assert abs(vols[0].item() - 5.0) < 1e-4   # 3-4-5 triangle hypotenuse
        assert abs(vols[1].item() - 1.0) < 1e-4

    def test_empty_simplex_list(self):
        pts = np.array([[0.0], [1.0]])
        sq = make_sq_dist(pts)
        edges = torch.zeros((0, 2), dtype=torch.long)
        vols = cayley_menger_volumes(edges, sq)
        assert vols.shape == (0,)

    def test_list_input(self):
        """Accepts plain Python list of tuples."""
        pts = np.array([[0.0, 0.0], [0.0, 1.0]])
        sq = make_sq_dist(pts)
        vols = cayley_menger_volumes([(0, 1)], sq)
        assert abs(vols[0].item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# cayley_menger_volumes — triangles
# ---------------------------------------------------------------------------

class TestCayleyMengerTriangles:
    """For a 2-simplex (triangle), vol = area."""

    def test_right_triangle(self):
        # Right triangle with legs 3 and 4 → area = 6
        pts = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        sq = make_sq_dist(pts)
        tris = torch.tensor([[0, 1, 2]])
        vol = cayley_menger_volumes(tris, sq)
        assert abs(vol.item() - 6.0) < 1e-4

    def test_equilateral_triangle(self):
        # Equilateral with side 2 → area = sqrt(3)
        import math
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, math.sqrt(3)]])
        sq = make_sq_dist(pts)
        tris = torch.tensor([[0, 1, 2]])
        vol = cayley_menger_volumes(tris, sq)
        expected = math.sqrt(3)
        assert abs(vol.item() - expected) < 1e-4

    def test_degenerate_triangle_clamped(self):
        """Collinear points → area ≈ 0, clamped to 1e-10."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        sq = make_sq_dist(pts)
        tris = torch.tensor([[0, 1, 2]])
        vol = cayley_menger_volumes(tris, sq)
        assert vol.item() >= 1e-10
        assert vol.item() < 1e-5


# ---------------------------------------------------------------------------
# cayley_menger_volumes — differentiability
# ---------------------------------------------------------------------------

class TestCayleyMengerDifferentiability:
    def test_gradient_flows(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        sq = make_sq_dist(pts).requires_grad_(True)
        tris = torch.tensor([[0, 1, 2]])
        vol = cayley_menger_volumes(tris, sq)
        vol.sum().backward()
        assert sq.grad is not None
        assert not torch.isnan(sq.grad).any()

    def test_gradient_flows_edges(self):
        pts = np.array([[0.0], [1.0], [3.0]])
        sq = make_sq_dist(pts).requires_grad_(True)
        edges = torch.tensor([[0, 1], [1, 2]])
        vols = cayley_menger_volumes(edges, sq)
        vols.sum().backward()
        assert sq.grad is not None
        assert not torch.isnan(sq.grad).any()


# ---------------------------------------------------------------------------
# geometric_hodge_laplacian_torch — basic properties
# ---------------------------------------------------------------------------

class TestGeometricHodgeLaplacianTorch:
    def _make_simple_complex(self, n_pts=8, seed=0):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((n_pts, 2))
        sc = SimplicialComplex(pts, max_dim=2)
        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)
        B1d = torch.tensor(B1.toarray(), dtype=torch.float32)
        B2d = torch.tensor(B2.toarray(), dtype=torch.float32) if B2 is not None else torch.zeros(B1d.shape[1], 0)
        n_edges = B1d.shape[1]
        n_tris = B2d.shape[1]
        # Unit volumes — combinatorial case
        v0 = torch.ones(n_pts)
        v1 = torch.ones(n_edges)
        v2 = torch.ones(n_tris) if n_tris > 0 else torch.ones(1)
        return B1d, B2d, v0, v1, v2, sc

    def test_combinatorial_recovery_with_unit_volumes(self):
        """With all volumes=1, should match combinatorial B₁ᵀB₁ + B₂B₂ᵀ."""
        B1d, B2d, v0, v1, v2, sc = self._make_simple_complex()
        if B1d.shape[1] == 0:
            pytest.skip("Empty complex")
        delta = geometric_hodge_laplacian_torch(B1d, B2d, v0, v1, v2)
        expected = B1d.T @ B1d + B2d @ B2d.T
        assert torch.allclose(delta, expected, atol=1e-5), \
            f"Max diff: {(delta - expected).abs().max().item():.2e}"

    def test_shape(self):
        B1d, B2d, v0, v1, v2, sc = self._make_simple_complex()
        if B1d.shape[1] == 0:
            pytest.skip("Empty complex")
        n_edges = B1d.shape[1]
        delta = geometric_hodge_laplacian_torch(B1d, B2d, v0, v1, v2)
        assert delta.shape == (n_edges, n_edges)

    def test_psd_symmetric_with_unit_volumes(self):
        """Combinatorial Hodge Laplacian must be PSD (eigenvalues >= 0)."""
        B1d, B2d, v0, v1, v2, sc = self._make_simple_complex()
        if B1d.shape[1] == 0:
            pytest.skip("Empty complex")
        delta = geometric_hodge_laplacian_torch(B1d, B2d, v0, v1, v2).detach().numpy()
        sym = 0.5 * (delta + delta.T)
        eigvals = eigh(sym, eigvals_only=True)
        assert eigvals.min() >= -1e-6, f"Min eigenvalue: {eigvals.min():.4e}"

    def test_none_upper_falls_back(self):
        """No upper boundary (B_kp1=None) → lower part only (unit volumes)."""
        B1d, _, v0, v1, _, sc = self._make_simple_complex()
        if B1d.shape[1] == 0:
            pytest.skip("Empty complex")
        # With unit v0 and v1: lower = B1d.T @ diag(1/v0) @ B1d @ diag(v1) = B1d.T @ B1d
        delta = geometric_hodge_laplacian_torch(B1d, None, v0, v1, None)
        expected = B1d.T @ B1d
        assert torch.allclose(delta, expected, atol=1e-5)

    def test_none_lower_falls_back(self):
        """No lower boundary (B_k=None) → upper part only (unit volumes).
        Call: geometric_hodge_laplacian_torch(None, B2d, None, v_edges, v_tris)
        Upper: M_k^{-1} B_{k+1} M_{k+1} B_{k+1}^T where k-simplices = edges.
        """
        B1d, B2d, _, v1, v2, sc = self._make_simple_complex()
        if B2d.shape[1] == 0:
            pytest.skip("No triangles")
        # v_k = v1 (edge vols, unit), v_kp1 = v2 (triangle vols, unit)
        delta = geometric_hodge_laplacian_torch(None, B2d, None, v1, v2)
        expected = B2d @ B2d.T
        assert torch.allclose(delta, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Diffusion distance computation
# ---------------------------------------------------------------------------

class TestDiffusionDistances:
    def _compute_diff_sq_dists(self, pts, steps=1, sigma=None):
        """Reimplement exactly as in graph_learning.py."""
        pts_t = torch.tensor(pts, dtype=torch.float32)
        diff2 = ((pts_t.unsqueeze(0) - pts_t.unsqueeze(1)) ** 2).sum(-1)
        if sigma is None:
            flat = diff2[diff2 > 0]
            sigma = float(flat.median()) if flat.numel() > 0 else 1.0
        W = torch.exp(-diff2 / sigma)
        d = W.sum(dim=1, keepdim=True).clamp_min(1e-8)
        P = W / d
        Pt = torch.linalg.matrix_power(P, steps)
        sq_diff_dists = ((Pt.unsqueeze(0) - Pt.unsqueeze(1)) ** 2).sum(-1)
        return sq_diff_dists

    def test_symmetric(self):
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((20, 3))
        sq = self._compute_diff_sq_dists(pts)
        assert torch.allclose(sq, sq.T, atol=1e-6)

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        pts = rng.standard_normal((20, 3))
        sq = self._compute_diff_sq_dists(pts)
        assert sq.min().item() >= -1e-6

    def test_zero_diagonal(self):
        rng = np.random.default_rng(2)
        pts = rng.standard_normal((20, 3))
        sq = self._compute_diff_sq_dists(pts)
        assert torch.allclose(sq.diag(), torch.zeros(20), atol=1e-6)

    def test_differentiable_through_matrix_power(self):
        """Grad should flow from sq_dists back to W (and therefore to inputs)."""
        pts = torch.tensor(np.random.default_rng(3).standard_normal((10, 2)),
                           dtype=torch.float32, requires_grad=True)
        diff2 = ((pts.unsqueeze(0) - pts.unsqueeze(1)) ** 2).sum(-1)
        W = torch.exp(-diff2)
        P = W / W.sum(1, keepdim=True).clamp_min(1e-8)
        Pt = torch.linalg.matrix_power(P, 2)
        sq = ((Pt.unsqueeze(0) - Pt.unsqueeze(1)) ** 2).sum(-1)
        sq.sum().backward()
        assert pts.grad is not None
        assert not torch.isnan(pts.grad).any()


# ---------------------------------------------------------------------------
# SWT geometric transition matrices
# ---------------------------------------------------------------------------

class TestSWTGeometricTransition:
    def _build_swt(self, n_pts=15, use_geom=False, seed=0):
        adj, pts = random_adj(n_pts, seed=seed)
        ro = torch.tensor(pts, dtype=torch.float32)
        sq = make_sq_dist(pts) if use_geom else None
        swt = SimplicialWaveletTransform(
            adj, ro, threshold=0.5, device="cpu",
            use_geometric_laplacian=use_geom,
            sq_diff_dists=sq,
        )
        return swt

    def test_combinatorial_P_set(self):
        swt = self._build_swt(use_geom=False)
        # At least P_U[0] or P_L[1] should be non-None
        has_upper = any(p is not None for p in swt.P_U)
        has_lower = any(p is not None for p in swt.P_L)
        assert has_upper or has_lower

    def test_geometric_P_set(self):
        swt = self._build_swt(use_geom=True)
        has_upper = any(p is not None for p in swt.P_U)
        has_lower = any(p is not None for p in swt.P_L)
        assert has_upper or has_lower

    def test_geometric_P_shapes(self):
        swt = self._build_swt(n_pts=15, use_geom=True)
        n_edges = swt.B[1].shape[1]
        if swt.P_U[0] is not None:
            assert swt.P_U[0].shape[0] == swt.B[1].shape[0], \
                "P_U[0] row dim should match node count"
        if swt.P_L[1] is not None:
            assert swt.P_L[1].shape == (n_edges, n_edges), \
                f"P_L[1] shape {swt.P_L[1].shape} != ({n_edges},{n_edges})"

    def test_wavelet_coeff_computable_combinatorial(self):
        swt = self._build_swt(use_geom=False)
        coeff = swt.calculate_wavelet_coeff(J=2)
        assert coeff.ndim == 1
        assert not torch.isnan(coeff).any()
        assert not torch.isinf(coeff).any()

    def test_wavelet_coeff_computable_geometric(self):
        swt = self._build_swt(use_geom=True)
        coeff = swt.calculate_wavelet_coeff(J=2)
        assert coeff.ndim == 1
        assert not torch.isnan(coeff).any()
        assert not torch.isinf(coeff).any()

    def test_geometric_coeff_different_from_combinatorial(self):
        """Geometric and combinatorial wavelet coefficients should differ."""
        adj, pts = random_adj(15, seed=7)
        ro = torch.tensor(pts, dtype=torch.float32)
        sq = make_sq_dist(pts)
        swt_c = SimplicialWaveletTransform(adj, ro, threshold=0.5, device="cpu",
                                           use_geometric_laplacian=False)
        swt_g = SimplicialWaveletTransform(adj, ro, threshold=0.5, device="cpu",
                                           use_geometric_laplacian=True,
                                           sq_diff_dists=sq)
        c_c = swt_c.calculate_wavelet_coeff(J=2)
        c_g = swt_g.calculate_wavelet_coeff(J=2)
        # They should not be identical (geometry changes the Laplacian)
        assert not torch.allclose(c_c, c_g, atol=1e-4), \
            "Geometric and combinatorial coefficients should differ"


class TestSparseBoundedSWT:
    def test_sparse_geometric_operators_are_bounded(self):
        torch.manual_seed(7)
        points = torch.randn(48, 4, requires_grad=True)
        layer = SimplicialFeatLearningLayer(
            n_weights=1,
            dimension=4,
            threshold=0.05,
            sigma=4.0,
            J=2,
            device="cpu",
            use_geometric_laplacian=True,
            max_neighbors=8,
            max_triangles=20,
        )
        distances = compute_dist(points)
        weights = torch.exp(-distances / layer.sigma)
        adjacency = torch.where(
            weights >= layer.threshold, weights, torch.zeros_like(weights)
        )
        adjacency = layer._sparsify_gaussian_graph(adjacency)
        transition = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-8)
        sq_diff_dists = compute_dist(transition)
        swt = SimplicialWaveletTransform(
            adjacency,
            points,
            threshold=layer.threshold,
            device="cpu",
            use_geometric_laplacian=True,
            sq_diff_dists=sq_diff_dists,
            max_triangles=20,
            sparse_operators=True,
            max_neighbors=8,
        )

        assert swt.B[1].is_sparse
        assert swt.B[2].is_sparse
        assert swt.edges.shape[0] <= points.shape[0] * 8 // 2
        assert swt.triangles.shape[0] <= 20
        assert hasattr(swt.P_L[1], "matmul")

        coeff = swt.calculate_wavelet_coeff(J=2)
        assert torch.isfinite(coeff).all()
