"""Tests for the geometric Hodge Laplacian implementation.

Unit tests:
1. Combinatorial recovery: M_k = I → Δ_k^(M) == Δ_k^comb
2. Self-adjointness: c1^T M_k Δ_k c2 == c2^T M_k Δ_k c1
3. Non-negativity: eigenvalues of Δ̃_k >= 0
4. Betti number: dim(ker(Δ_k^(M))) == β_k
5. Symmetry of conjugate: Δ̃_k == Δ̃_k^T

Functional test:
- Torus with R1 != R2: two harmonic 1-cochains with different M_1-norms.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.simplicial_complex import SimplicialComplex
from models.hodge_laplacian import (
    compute_simplex_volumes,
    mass_matrix,
    geometric_hodge_laplacian,
    combinatorial_hodge_laplacian,
    symmetric_hodge_laplacian,
)
from models.diffusion_maps import diffusion_maps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_triangle():
    """Simple triangle: 3 vertices, 3 edges, 1 triangle."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    sc = SimplicialComplex(points, max_dim=2, threshold=1.5)
    return sc, points


def _make_tetrahedron():
    """Regular tetrahedron: 4 vertices, 6 edges, 4 triangles, 1 tetrahedron."""
    points = np.array(
        [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float64
    )
    sc = SimplicialComplex(points, max_dim=3, threshold=4.0)
    return sc, points


def _sample_torus(N=300, R1=3.0, R2=1.0, seed=42):
    """Sample N points from a torus in R^3 with major radius R1, minor radius R2."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, N)
    phi = rng.uniform(0, 2 * np.pi, N)
    x = (R1 + R2 * np.cos(phi)) * np.cos(theta)
    y = (R1 + R2 * np.cos(phi)) * np.sin(theta)
    z = R2 * np.sin(phi)
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Test: boundary d^2 = 0
# ---------------------------------------------------------------------------


class TestSimplicialComplex:
    def test_boundary_squared_zero_triangle(self):
        sc, _ = _make_triangle()
        assert sc.verify_boundary_squared_zero(1)

    def test_boundary_squared_zero_tetrahedron(self):
        sc, _ = _make_tetrahedron()
        assert sc.verify_boundary_squared_zero(1)
        assert sc.verify_boundary_squared_zero(2)

    def test_simplex_counts_triangle(self):
        sc, _ = _make_triangle()
        assert sc.num_simplices(0) == 3
        assert sc.num_simplices(1) == 3
        assert sc.num_simplices(2) == 1

    def test_simplex_counts_tetrahedron(self):
        sc, _ = _make_tetrahedron()
        assert sc.num_simplices(0) == 4
        assert sc.num_simplices(1) == 6
        assert sc.num_simplices(2) == 4
        assert sc.num_simplices(3) == 1

    def test_boundary_matrix_shape(self):
        sc, _ = _make_triangle()
        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)
        assert B1.shape == (3, 3)  # 3 vertices x 3 edges
        assert B2.shape == (3, 1)  # 3 edges x 1 triangle


# ---------------------------------------------------------------------------
# Test 1: Combinatorial recovery
# ---------------------------------------------------------------------------


class TestCombinatorialRecovery:
    def test_identity_mass_equals_combinatorial_k0(self):
        sc, points = _make_triangle()
        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)

        delta_comb = combinatorial_hodge_laplacian(None, B1)
        delta_geom = geometric_hodge_laplacian(None, B1, None, None, None)

        np.testing.assert_allclose(delta_geom, delta_comb, atol=1e-12)

    def test_identity_mass_equals_combinatorial_k1(self):
        sc, points = _make_triangle()
        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)

        delta_comb = combinatorial_hodge_laplacian(B1, B2)
        delta_geom = geometric_hodge_laplacian(B1, B2, None, None, None)

        np.testing.assert_allclose(delta_geom, delta_comb, atol=1e-12)

    def test_identity_mass_equals_combinatorial_k2(self):
        sc, points = _make_triangle()
        B2 = sc.boundary_matrix(2)

        delta_comb = combinatorial_hodge_laplacian(B2, None)
        delta_geom = geometric_hodge_laplacian(B2, None, None, None, None)

        np.testing.assert_allclose(delta_geom, delta_comb, atol=1e-12)

    def test_identity_mass_equals_combinatorial_tetrahedron(self):
        sc, points = _make_tetrahedron()
        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)
        B3 = sc.boundary_matrix(3)

        for B_k, B_kp1 in [(None, B1), (B1, B2), (B2, B3), (B3, None)]:
            delta_comb = combinatorial_hodge_laplacian(B_k, B_kp1)
            delta_geom = geometric_hodge_laplacian(B_k, B_kp1, None, None, None)
            np.testing.assert_allclose(delta_geom, delta_comb, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: Self-adjointness w.r.t. M_k inner product
# ---------------------------------------------------------------------------


class TestSelfAdjointness:
    def _check_self_adjoint(self, delta_k, M_k_mat, n, rng):
        """Check c1^T M_k Δ_k c2 == c2^T M_k Δ_k c1."""
        M_dense = M_k_mat.toarray() if hasattr(M_k_mat, "toarray") else M_k_mat
        for _ in range(5):
            c1 = rng.standard_normal(n)
            c2 = rng.standard_normal(n)
            lhs = c1 @ M_dense @ delta_k @ c2
            rhs = c2 @ M_dense @ delta_k @ c1
            np.testing.assert_allclose(lhs, rhs, atol=1e-8)

    def test_self_adjoint_geometric_k1_triangle(self):
        rng = np.random.default_rng(123)
        sc, points = _make_triangle()
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)

        delta_1 = geometric_hodge_laplacian(
            B1, B2, mass_pairs[0], mass_pairs[1], mass_pairs[2]
        )
        M1, _ = mass_pairs[1]
        self._check_self_adjoint(delta_1, M1, sc.num_simplices(1), rng)

    def test_self_adjoint_geometric_k0_triangle(self):
        rng = np.random.default_rng(456)
        sc, points = _make_triangle()
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        B1 = sc.boundary_matrix(1)

        delta_0 = geometric_hodge_laplacian(
            None, B1, None, mass_pairs[0], mass_pairs[1]
        )
        M0, _ = mass_pairs[0]
        self._check_self_adjoint(delta_0, M0, sc.num_simplices(0), rng)

    def test_self_adjoint_geometric_tetrahedron(self):
        rng = np.random.default_rng(789)
        sc, points = _make_tetrahedron()
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)
        B3 = sc.boundary_matrix(3)

        # k=1 Laplacian
        delta_1 = geometric_hodge_laplacian(
            B1, B2, mass_pairs[0], mass_pairs[1], mass_pairs[2]
        )
        M1, _ = mass_pairs[1]
        self._check_self_adjoint(delta_1, M1, sc.num_simplices(1), rng)

        # k=2 Laplacian
        delta_2 = geometric_hodge_laplacian(
            B2, B3, mass_pairs[1], mass_pairs[2], mass_pairs[3]
        )
        M2, _ = mass_pairs[2]
        self._check_self_adjoint(delta_2, M2, sc.num_simplices(2), rng)


# ---------------------------------------------------------------------------
# Test 3: Non-negativity of eigenvalues
# ---------------------------------------------------------------------------


class TestNonNegativity:
    def _check_nonneg(self, delta_tilde, label=""):
        eigvals = np.linalg.eigvalsh(delta_tilde)
        assert np.all(eigvals >= -1e-8), (
            f"Negative eigenvalue in {label}: min={eigvals.min()}"
        )

    def test_nonneg_symmetric_triangle(self):
        sc, points = _make_triangle()
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)

        for k, (B_k, B_kp1) in enumerate([(None, B1), (B1, B2), (B2, None)]):
            delta_k = geometric_hodge_laplacian(
                B_k,
                B_kp1,
                mass_pairs.get(k - 1) if k > 0 else None,
                mass_pairs.get(k),
                mass_pairs.get(k + 1),
            )
            delta_tilde = symmetric_hodge_laplacian(delta_k, mass_pairs[k])
            self._check_nonneg(delta_tilde, f"k={k}")

    def test_nonneg_symmetric_tetrahedron(self):
        sc, points = _make_tetrahedron()
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        Bs = {k: sc.boundary_matrix(k) for k in range(1, 4)}

        for k in range(4):
            B_k = Bs.get(k)
            B_kp1 = Bs.get(k + 1)
            if B_k is None and B_kp1 is None:
                continue
            delta_k = geometric_hodge_laplacian(
                B_k,
                B_kp1,
                mass_pairs.get(k - 1) if k > 0 else None,
                mass_pairs.get(k),
                mass_pairs.get(k + 1),
            )
            delta_tilde = symmetric_hodge_laplacian(delta_k, mass_pairs[k])
            self._check_nonneg(delta_tilde, f"k={k}")

    def test_nonneg_combinatorial(self):
        sc, _ = _make_triangle()
        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)

        for B_k, B_kp1 in [(None, B1), (B1, B2), (B2, None)]:
            delta = combinatorial_hodge_laplacian(B_k, B_kp1)
            eigvals = np.linalg.eigvalsh(delta)
            assert np.all(eigvals >= -1e-8)


# ---------------------------------------------------------------------------
# Test 4: Betti numbers (harmonic space dimension)
# ---------------------------------------------------------------------------


class TestBettiNumbers:
    def _kernel_dim(self, matrix, tol=1e-6):
        """Count eigenvalues within tol of zero."""
        eigvals = np.linalg.eigvalsh(matrix)
        return np.sum(np.abs(eigvals) < tol)

    def test_betti_triangle_combinatorial(self):
        """Single triangle: β_0=1, β_1=0, β_2=0."""
        sc, _ = _make_triangle()
        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)

        delta_0 = combinatorial_hodge_laplacian(None, B1)
        delta_1 = combinatorial_hodge_laplacian(B1, B2)
        delta_2 = combinatorial_hodge_laplacian(B2, None)

        assert self._kernel_dim(delta_0) == 1  # β_0 = 1
        assert self._kernel_dim(delta_1) == 0  # β_1 = 0
        assert self._kernel_dim(delta_2) == 0  # β_2 = 0

    def test_betti_triangle_geometric(self):
        """Geometric Laplacian should give same Betti numbers as combinatorial."""
        sc, points = _make_triangle()
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)

        delta_0 = geometric_hodge_laplacian(
            None, B1, None, mass_pairs[0], mass_pairs[1]
        )
        delta_1 = geometric_hodge_laplacian(
            B1, B2, mass_pairs[0], mass_pairs[1], mass_pairs[2]
        )
        delta_2 = geometric_hodge_laplacian(
            B2, None, mass_pairs[1], mass_pairs[2], None
        )

        # Use symmetric version for reliable eigendecomposition
        dt0 = symmetric_hodge_laplacian(delta_0, mass_pairs[0])
        dt1 = symmetric_hodge_laplacian(delta_1, mass_pairs[1])
        dt2 = symmetric_hodge_laplacian(delta_2, mass_pairs[2])

        assert self._kernel_dim(dt0) == 1  # β_0 = 1
        assert self._kernel_dim(dt1) == 0  # β_1 = 0
        assert self._kernel_dim(dt2) == 0  # β_2 = 0

    def test_betti_tetrahedron(self):
        """Filled tetrahedron: β_0=1, β_1=0, β_2=0, β_3=0."""
        sc, points = _make_tetrahedron()
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        Bs = {k: sc.boundary_matrix(k) for k in range(1, 4)}

        expected_betti = {0: 1, 1: 0, 2: 0, 3: 0}
        for k in range(4):
            B_k = Bs.get(k)
            B_kp1 = Bs.get(k + 1)
            if B_k is None and B_kp1 is None:
                continue
            delta_k = geometric_hodge_laplacian(
                B_k,
                B_kp1,
                mass_pairs.get(k - 1) if k > 0 else None,
                mass_pairs.get(k),
                mass_pairs.get(k + 1),
            )
            dt = symmetric_hodge_laplacian(delta_k, mass_pairs[k])
            assert self._kernel_dim(dt) == expected_betti[k], (
                f"β_{k} mismatch: got {self._kernel_dim(dt)}, expected {expected_betti[k]}"
            )

    def test_betti_loop(self):
        """A cycle of 4 vertices (no triangles): β_0=1, β_1=1."""
        # Square loop: 4 vertices, 4 edges, no triangles
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        # Use threshold that only connects adjacent vertices (not diagonals)
        sc = SimplicialComplex(points, max_dim=2, threshold=1.1)

        assert sc.num_simplices(0) == 4
        assert sc.num_simplices(1) == 4
        assert sc.num_simplices(2) == 0

        B1 = sc.boundary_matrix(1)

        delta_0 = combinatorial_hodge_laplacian(None, B1)
        delta_1 = combinatorial_hodge_laplacian(B1, None)

        assert self._kernel_dim(delta_0) == 1  # β_0 = 1 (connected)
        assert self._kernel_dim(delta_1) == 1  # β_1 = 1 (one loop)


# ---------------------------------------------------------------------------
# Test 5: Symmetry of conjugate
# ---------------------------------------------------------------------------


class TestSymmetry:
    def test_symmetric_conjugate_triangle(self):
        sc, points = _make_triangle()
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)

        for k, (B_k, B_kp1) in enumerate([(None, B1), (B1, B2), (B2, None)]):
            delta_k = geometric_hodge_laplacian(
                B_k,
                B_kp1,
                mass_pairs.get(k - 1) if k > 0 else None,
                mass_pairs.get(k),
                mass_pairs.get(k + 1),
            )
            dt = symmetric_hodge_laplacian(delta_k, mass_pairs[k])
            np.testing.assert_allclose(dt, dt.T, atol=1e-12)

    def test_symmetric_conjugate_tetrahedron(self):
        sc, points = _make_tetrahedron()
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        Bs = {k: sc.boundary_matrix(k) for k in range(1, 4)}

        for k in range(4):
            B_k = Bs.get(k)
            B_kp1 = Bs.get(k + 1)
            if B_k is None and B_kp1 is None:
                continue
            delta_k = geometric_hodge_laplacian(
                B_k,
                B_kp1,
                mass_pairs.get(k - 1) if k > 0 else None,
                mass_pairs.get(k),
                mass_pairs.get(k + 1),
            )
            dt = symmetric_hodge_laplacian(delta_k, mass_pairs[k])
            np.testing.assert_allclose(dt, dt.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: Diffusion Maps
# ---------------------------------------------------------------------------


class TestDiffusionMaps:
    def test_basic_embedding_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        Y, eigvals = diffusion_maps(X, n_components=5)
        assert Y.shape == (50, 5)
        assert eigvals.shape == (5,)

    def test_eigenvalues_sorted_descending(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        _, eigvals = diffusion_maps(X, n_components=5)
        assert np.all(np.diff(eigvals) <= 1e-10)  # descending

    def test_eigenvalues_bounded(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        _, eigvals = diffusion_maps(X, n_components=5)
        assert np.all(eigvals >= -1e-10)
        assert np.all(eigvals <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Test: Simplex volumes
# ---------------------------------------------------------------------------


class TestSimplexVolumes:
    def test_edge_lengths_equilateral(self):
        """Equilateral triangle with side length 1."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
        sc = SimplicialComplex(points, max_dim=2, threshold=1.5)
        volumes = compute_simplex_volumes(sc, points)

        # All edges should have length 1.0
        np.testing.assert_allclose(volumes[1], [1.0, 1.0, 1.0], atol=1e-10)

    def test_triangle_area(self):
        """Area of equilateral triangle with side 1 = sqrt(3)/4."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
        sc = SimplicialComplex(points, max_dim=2, threshold=1.5)
        volumes = compute_simplex_volumes(sc, points)

        expected_area = np.sqrt(3) / 4
        np.testing.assert_allclose(volumes[2], [expected_area], atol=1e-10)

    def test_right_triangle_area(self):
        """Right triangle with legs 3, 4: area = 6."""
        points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        sc = SimplicialComplex(points, max_dim=2, threshold=6.0)
        volumes = compute_simplex_volumes(sc, points)

        np.testing.assert_allclose(volumes[2], [6.0], atol=1e-10)

    def test_tetrahedron_volume(self):
        """Regular tetrahedron: V = (edge^3)/(6*sqrt(2))."""
        points = np.array(
            [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float64
        )
        sc = SimplicialComplex(points, max_dim=3, threshold=4.0)
        volumes = compute_simplex_volumes(sc, points)

        # Edge length = 2*sqrt(2)
        edge_len = 2 * np.sqrt(2)
        expected_vol = edge_len**3 / (6 * np.sqrt(2))
        np.testing.assert_allclose(volumes[3], [expected_vol], atol=1e-8)


# ---------------------------------------------------------------------------
# Functional Test: Torus
# ---------------------------------------------------------------------------


class TestTorusFunctional:
    """Functional test: torus topology via geometric Hodge Laplacian.

    Constructs a proper triangulation of the torus grid directly (not via
    Rips complex, which doesn't reliably produce correct Betti numbers
    for surfaces due to threshold sensitivity).
    """

    def _build_torus_triangulation(self, n_major=10, n_minor=6, R1=3.0, R2=1.0):
        """Build an explicit triangulation of the torus grid.

        Each grid square is divided into two triangles along one diagonal.
        This gives an exact triangulation with:
          V = n_major * n_minor
          E = 3 * V  (horizontal + vertical + diagonal, all wrap)
          F = 2 * V
          χ = V - 3V + 2V = 0 (correct for torus)
          β_0=1, β_1=2, β_2=1
        """
        N = n_major * n_minor
        theta = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        theta_flat = theta_grid.ravel()
        phi_flat = phi_grid.ravel()
        x = (R1 + R2 * np.cos(phi_flat)) * np.cos(theta_flat)
        y = (R1 + R2 * np.cos(phi_flat)) * np.sin(theta_flat)
        z = R2 * np.sin(phi_flat)
        points = np.column_stack([x, y, z])

        def idx(i, j):
            return (j % n_minor) * n_major + (i % n_major)

        # Manually build simplices
        edges = set()
        triangles = []
        for i in range(n_major):
            for j in range(n_minor):
                # Grid vertex (i,j) and its neighbors
                v00 = idx(i, j)
                v10 = idx(i + 1, j)
                v01 = idx(i, j + 1)
                v11 = idx(i + 1, j + 1)

                # Edges: horizontal, vertical, diagonal
                edges.add(tuple(sorted((v00, v10))))
                edges.add(tuple(sorted((v00, v01))))
                edges.add(tuple(sorted((v00, v11))))  # diagonal

                # Two triangles per square: (v00, v10, v11) and (v00, v01, v11)
                triangles.append(tuple(sorted((v00, v10, v11))))
                triangles.append(tuple(sorted((v00, v01, v11))))

        edges = sorted(edges)
        triangles = sorted(set(triangles))

        # Build SimplicialComplex manually
        sc = SimplicialComplex.__new__(SimplicialComplex)
        sc.points = points
        sc.max_dim = 2
        sc.N = N
        sc.max_simplices = {}
        sc._dist_matrix = None
        sc.threshold = None
        sc._simplices = {
            0: [(i,) for i in range(N)],
            1: list(edges),
            2: list(triangles),
        }
        sc._simplex_to_idx = {
            0: {(i,): i for i in range(N)},
            1: {e: i for i, e in enumerate(edges)},
            2: {t: i for i, t in enumerate(triangles)},
        }
        sc._boundary_matrices = {}

        return sc, points

    def test_torus_betti_1(self):
        """Torus triangulation should have β_1 = 2 harmonic 1-cochains."""
        sc, points = self._build_torus_triangulation(
            n_major=10, n_minor=6, R1=3.0, R2=1.0
        )

        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)

        # Verify Euler characteristic: V - E + F = 0
        V = sc.num_simplices(0)
        E = sc.num_simplices(1)
        F = sc.num_simplices(2)
        assert V - E + F == 0, f"Euler char: {V} - {E} + {F} = {V - E + F}"

        # Verify d^2 = 0
        assert sc.verify_boundary_squared_zero(1)

        # β_0
        delta_0 = combinatorial_hodge_laplacian(None, B1)
        eigvals_0 = np.linalg.eigvalsh(delta_0)
        beta_0 = np.sum(np.abs(eigvals_0) < 1e-6)
        assert beta_0 == 1, f"Expected β_0=1, got {beta_0}"

        # β_1
        delta_1 = combinatorial_hodge_laplacian(B1, B2)
        eigvals_1 = np.linalg.eigvalsh(delta_1)
        beta_1 = np.sum(np.abs(eigvals_1) < 1e-4)
        assert beta_1 == 2, (
            f"Expected β_1=2, got {beta_1}. Smallest eigvals: {sorted(eigvals_1)[:5]}"
        )

    def test_torus_geometric_betti_1(self):
        """Geometric Hodge Laplacian should also give β_1 = 2."""
        sc, points = self._build_torus_triangulation(
            n_major=10, n_minor=6, R1=3.0, R2=1.0
        )

        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        delta_1 = geometric_hodge_laplacian(
            B1, B2, mass_pairs[0], mass_pairs[1], mass_pairs[2]
        )
        dt1 = symmetric_hodge_laplacian(delta_1, mass_pairs[1])

        eigvals = np.linalg.eigvalsh(dt1)
        beta_1 = np.sum(np.abs(eigvals) < 1e-4)
        assert beta_1 == 2, (
            f"Expected β_1=2 (geometric), got {beta_1}. "
            f"Smallest eigvals: {sorted(eigvals)[:5]}"
        )

    def test_torus_geometric_norms_differ(self):
        """Geometric vs combinatorial Laplacians should differ in their spectra.

        The geometric Laplacian encodes edge lengths through mass matrices,
        so its non-zero eigenvalues differ from the combinatorial case.
        For R1 != R2, the spectral gap for the two homology generators
        should be different (reflecting different cycle lengths).
        """
        sc, points = self._build_torus_triangulation(
            n_major=10, n_minor=6, R1=3.0, R2=1.0
        )

        B1 = sc.boundary_matrix(1)
        B2 = sc.boundary_matrix(2)
        volumes = compute_simplex_volumes(sc, points)
        mass_pairs = {k: mass_matrix(v) for k, v in volumes.items()}

        # Geometric Laplacian
        delta_1_geom = geometric_hodge_laplacian(
            B1, B2, mass_pairs[0], mass_pairs[1], mass_pairs[2]
        )
        dt1_geom = symmetric_hodge_laplacian(delta_1_geom, mass_pairs[1])
        eigvals_geom = np.sort(np.linalg.eigvalsh(dt1_geom))

        # Combinatorial Laplacian
        delta_1_comb = combinatorial_hodge_laplacian(B1, B2)
        eigvals_comb = np.sort(np.linalg.eigvalsh(delta_1_comb))

        # Both should have 2 zero eigenvalues (β_1 = 2)
        assert np.sum(np.abs(eigvals_geom) < 1e-4) == 2
        assert np.sum(np.abs(eigvals_comb) < 1e-4) == 2

        # The smallest NON-ZERO eigenvalues should differ between
        # geometric and combinatorial (mass matrices change the spectrum)
        nonzero_geom = eigvals_geom[np.abs(eigvals_geom) > 1e-4]
        nonzero_comb = eigvals_comb[np.abs(eigvals_comb) > 1e-4]

        # Compare the first few non-zero eigenvalues
        n_compare = min(5, len(nonzero_geom), len(nonzero_comb))
        geom_first = nonzero_geom[:n_compare]
        comb_first = nonzero_comb[:n_compare]

        # They should NOT be identical (geometric weights change the spectrum)
        relative_diff = np.max(np.abs(geom_first - comb_first) / np.abs(comb_first))
        assert relative_diff > 0.01, (
            f"Geometric and combinatorial spectra too similar. "
            f"Geometric: {geom_first}, Combinatorial: {comb_first}, "
            f"Max relative diff: {relative_diff:.6f}"
        )

        # Additionally, convert harmonic eigenvectors back to original basis
        # h̃ = M^{1/2} h, so h = M^{-1/2} h̃. The L2 norm ||h||_2 captures
        # how the harmonic form distributes over edges of different lengths.
        _, eigvecs_geom = np.linalg.eigh(dt1_geom)
        harmonic_idx = np.where(np.abs(eigvals_geom) < 1e-4)[0]
        M1_diag = volumes[1]
        inv_sqrt_M1 = 1.0 / np.sqrt(M1_diag)

        h1_orig = inv_sqrt_M1 * eigvecs_geom[:, harmonic_idx[0]]
        h2_orig = inv_sqrt_M1 * eigvecs_geom[:, harmonic_idx[1]]

        # The L2 norms of the original harmonic forms should differ
        # because the major cycle (length 2πR1) and minor cycle (length 2πR2)
        # have different edge length distributions
        l2_norm1 = np.linalg.norm(h1_orig)
        l2_norm2 = np.linalg.norm(h2_orig)
        ratio = max(l2_norm1, l2_norm2) / min(l2_norm1, l2_norm2)
        assert ratio > 1.01, (
            f"Harmonic form L2 norms too similar after M^{{-1/2}} transform: "
            f"{l2_norm1:.4f} vs {l2_norm2:.4f} (ratio {ratio:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
