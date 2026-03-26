"""Shared utilities for building combinatorial and geometric Hodge 1-Laplacians.

All functions work in numpy/scipy for experiment scripts (no autograd needed).
Diffusion distances are computed from scratch here; the torch version in
graph_learning.py is used for training.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))  # project root

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

from models.simplicial_complex import SimplicialComplex
from models.hodge_laplacian import cayley_menger_volumes


# ---------------------------------------------------------------------------
# Diffusion distances
# ---------------------------------------------------------------------------

def compute_diffusion_sq_dists(points, sigma=None, steps=1):
    """Pairwise squared diffusion distances D²_t(i,j) = ||P^t[i,:] - P^t[j,:]||².

    Parameters
    ----------
    points : np.ndarray, shape (N, d)
    sigma : float or None  — Gaussian kernel bandwidth (auto = median sq-dist)
    steps : int            — diffusion time t

    Returns
    -------
    sq_dists : np.ndarray, shape (N, N)
    """
    D_sq = squareform(pdist(points)) ** 2
    if sigma is None:
        flat = D_sq[D_sq > 0]
        sigma = float(np.median(flat)) if len(flat) > 0 else 1.0
    W = np.exp(-D_sq / sigma)
    row_sums = W.sum(axis=1, keepdims=True).clip(min=1e-12)
    P = W / row_sums
    Pt = np.linalg.matrix_power(P, steps)
    return squareform(pdist(Pt, 'sqeuclidean'))  # O(N²) memory


# ---------------------------------------------------------------------------
# Laplacian builders
# ---------------------------------------------------------------------------

def build_rips_laplacians(points, threshold=None, max_dim=2, sigma=None,
                          diffusion_steps=1, max_triangles=10_000, verbose=True):
    """Build combinatorial and (symmetric) geometric Hodge 1-Laplacians.

    Parameters
    ----------
    points : np.ndarray, shape (N, d)
    threshold : float or None  — Rips threshold (None = median pairwise dist)
    max_dim : int
    sigma : float or None      — DM kernel bandwidth
    diffusion_steps : int      — t for P^t
    max_triangles : int or None — cap on 2-simplices; keeps lowest-filtration ones (default 10k)
    verbose : bool             — print per-step timing logs

    Returns
    -------
    L1_comb : np.ndarray (n_edges, n_edges)  — combinatorial Δ₁ (symmetric)
    L1_geom : np.ndarray (n_edges, n_edges)  — symmetric conjugate Δ̃₁ = M₁^½ Δ₁ M₁^{-½}
    sc       : SimplicialComplex
    vols     : dict  with keys 'v0', 'v1', 'v2' (numpy volume arrays)
    B1d      : np.ndarray (n_verts, n_edges)  — signed boundary matrix
    B2d      : np.ndarray (n_edges, n_tris)   — signed boundary matrix
    """
    import time
    def log(msg):
        if verbose:
            print(f"      [build_rips] {msg}", flush=True)

    t0 = time.time()
    max_simplices = {2: max_triangles} if max_triangles is not None else {}
    log(f"SimplicialComplex(N={len(points)}, max_dim={max_dim}, threshold={threshold}, "
        f"max_triangles={max_triangles}) ...")
    sc = SimplicialComplex(points, max_dim=max_dim, threshold=threshold,
                           max_simplices=max_simplices)
    log(f"done in {time.time()-t0:.2f}s  |  threshold={sc.threshold:.4f}  |  "
        f"{sc.num_simplices(0)} verts  {sc.num_simplices(1)} edges  {sc.num_simplices(2)} tris")

    t1 = time.time()
    log("boundary_matrix(1) ...")
    B1 = sc.boundary_matrix(1)
    log(f"done in {time.time()-t1:.2f}s")

    t1 = time.time()
    log("boundary_matrix(2) ...")
    B2 = sc.boundary_matrix(2)
    log(f"done in {time.time()-t1:.2f}s")

    B1d = B1.toarray() if B1 is not None else np.zeros((sc.N, 0))
    B2d = B2.toarray() if B2 is not None else np.zeros((max(1, B1d.shape[1]), 0))

    n_edges = B1d.shape[1]
    empty = np.zeros((0, 0))

    if n_edges == 0:
        log("empty complex — returning zeros")
        return (empty, empty, sc,
                {"v0": np.array([]), "v1": np.array([]), "v2": np.array([])},
                B1d, B2d)

    # --- Combinatorial Δ₁ (already symmetric) ---
    t1 = time.time()
    log("combinatorial L1 = B1.T@B1 + B2@B2.T ...")
    L1_comb = B1d.T @ B1d + B2d @ B2d.T
    log(f"done in {time.time()-t1:.2f}s  |  shape={L1_comb.shape}")

    # --- Diffusion distances (numpy) ---
    t1 = time.time()
    log(f"compute_diffusion_sq_dists(steps={diffusion_steps}, sigma={sigma}) ...")
    sq_dists = compute_diffusion_sq_dists(points, sigma=sigma, steps=diffusion_steps)
    log(f"done in {time.time()-t1:.2f}s  |  "
        f"sq_dists range=[{sq_dists.min():.2e}, {sq_dists.max():.4f}]")
    sq_dists_t = torch.tensor(sq_dists, dtype=torch.float32)

    # Volumes via Cayley-Menger determinant
    n_verts = sc.N
    v0 = np.full(n_verts, 1.0 / n_verts)

    edges = sc.simplices.get(1, [])
    t1 = time.time()
    log(f"cayley_menger_volumes for {len(edges)} edges ...")
    if edges:
        v1 = cayley_menger_volumes(
            torch.tensor(edges, dtype=torch.long), sq_dists_t
        ).detach().numpy()
    else:
        v1 = np.ones(n_edges)
    log(f"done in {time.time()-t1:.2f}s  |  "
        f"v1 range=[{v1.min():.2e}, {v1.max():.4f}]  mean={v1.mean():.4f}")

    tris = sc.simplices.get(2, [])
    n_tris = B2d.shape[1]
    t1 = time.time()
    log(f"cayley_menger_volumes for {len(tris)} triangles ...")
    if tris:
        v2 = cayley_menger_volumes(
            torch.tensor(tris, dtype=torch.long), sq_dists_t
        ).detach().numpy()
    else:
        v2 = np.ones(max(n_tris, 1))
    log(f"done in {time.time()-t1:.2f}s  |  "
        f"v2 range=[{v2.min():.2e}, {v2.max():.4f}]  mean={v2.mean():.4f}")

    # --- Geometric Δ₁ = M₁⁻¹ B₂ M₂ B₂ᵀ + B₁ᵀ M₀⁻¹ B₁ M₁ ---
    t1 = time.time()
    log("geometric L1 assembly ...")
    L1_geom = np.zeros((n_edges, n_edges))
    if n_tris > 0:
        L1_geom += np.diag(1.0 / v1) @ B2d @ np.diag(v2[:n_tris]) @ B2d.T
    L1_geom += B1d.T @ np.diag(1.0 / v0) @ B1d @ np.diag(v1)
    log(f"done in {time.time()-t1:.2f}s")

    # Symmetrize: Δ̃₁ = M₁^½ Δ₁ M₁^{-½}
    t1 = time.time()
    log("symmetrizing geometric L1 ...")
    sqrt_v1 = np.sqrt(np.maximum(v1, 1e-20))
    inv_sqrt_v1 = 1.0 / sqrt_v1
    L1_geom_sym = (sqrt_v1[:, None] * L1_geom) * inv_sqrt_v1[None, :]
    L1_geom_sym = 0.5 * (L1_geom_sym + L1_geom_sym.T)
    log(f"done in {time.time()-t1:.2f}s  |  "
        f"sym_err={np.abs(L1_geom_sym - L1_geom_sym.T).max():.2e}  "
        f"range=[{L1_geom_sym.min():.4f}, {L1_geom_sym.max():.4f}]")

    log(f"total build time: {time.time()-t0:.2f}s")
    vols = {"v0": v0, "v1": v1, "v2": v2[:n_tris] if tris else np.ones(0)}
    return L1_comb, L1_geom_sym, sc, vols, B1d, B2d


# ---------------------------------------------------------------------------
# Spectral utilities
# ---------------------------------------------------------------------------

def _gap_split(eigvals, min_gap_ratio=10.0, max_search=30):
    """Find the index that splits harmonic (near-zero) from non-harmonic eigenvalues.

    Looks for the largest *relative* jump ev[i+1]/ev[i] among the first
    max_search positive eigenvalues. Restricting to the leading eigenvalues
    prevents spuriously large gaps deep in the tail (which can occur for the
    geometric Laplacian) from being mistaken for the harmonic boundary.

    Falls back to absolute tol 1e-6 if no clear gap is found.

    Returns the number of eigenvalues considered harmonic (i.e. β estimate).
    """
    ev = np.sort(np.maximum(eigvals, 0.0))
    pos = ev[ev > 1e-10]
    pos = pos[:max_search]   # only search leading eigenvalues

    if len(pos) < 2:
        return int(np.sum(ev <= 1e-6))

    ratios = pos[1:] / pos[:-1]
    best = int(np.argmax(ratios))
    if ratios[best] >= min_gap_ratio:
        return int(np.sum(ev <= pos[best]))
    else:
        return int(np.sum(ev <= 1e-6))


def get_betti_1(L):
    """Estimate β₁ as the number of near-zero eigenvalues using a gap-based split."""
    if L.shape[0] == 0:
        return 0
    eigvals = eigh(L, eigvals_only=True)
    return _gap_split(eigvals)


def harmonic_subspace(L, tol=1e-4):
    """Return orthonormal basis for null(L) (harmonic subspace)."""
    if L.shape[0] == 0:
        return np.zeros((0, 0))
    eigvals, eigvecs = eigh(L)
    return eigvecs[:, eigvals < tol]


def compute_eigenvalue_ratios(L, K=10, n_harm=None):
    """Normalized eigenvalue ratios λ_k / λ_1 for k=2..K+1.

    Uses _gap_split to find the spectral gap (separating harmonic from
    non-harmonic eigenvalues), then computes ratios from the non-harmonic part.
    This avoids picking up numerical noise eigenvalues (~1e-6) as "non-zero".

    Parameters
    ----------
    L : np.ndarray (n, n) — symmetric Laplacian matrix
    K : int — number of ratios to return
    n_harm : int or None — if provided, skip this many eigenvalues instead of
             using _gap_split (useful when β₁ is known a priori)

    Returns
    -------
    ratios : np.ndarray (K,) or None if not enough eigenvalues
    """
    if L.shape[0] == 0:
        return None
    eigvals = np.maximum(eigh(L, eigvals_only=True), 0.0)
    eigvals = np.sort(eigvals)
    if n_harm is None:
        n_harm = _gap_split(eigvals)
    nonzero = eigvals[n_harm:]
    # Also skip any residual near-zero eigenvalues above n_harm
    nonzero = nonzero[nonzero > 1e-4]
    if len(nonzero) < K + 1:
        return None
    lam1 = nonzero[0]
    return nonzero[1:K + 1] / lam1


def first_spectral_gap(L):
    """Normalized spectral gap: λ_{β₁+1} / λ_max.

    Normalizing by the largest eigenvalue makes this scale-invariant, so
    combinatorial and geometric Laplacians (which differ by ~300× in scale)
    can be compared on the same axis.
    """
    if L.shape[0] == 0:
        return 0.0
    eigvals = np.maximum(eigh(L, eigvals_only=True), 0.0)
    lam_max = eigvals.max()
    if lam_max < 1e-12:
        return 0.0
    n = _gap_split(eigvals)
    gap = float(eigvals[n]) if n < len(eigvals) else 0.0
    return gap / lam_max
