"""Point cloud samplers for synthetic manifolds with known topology.

Betti numbers:
  torus:  β₀=1, β₁=2, β₂=1
  sphere: β₀=1, β₁=0, β₂=1
"""

import numpy as np

BETTI_NUMBERS = {
    "torus":  {0: 1, 1: 2, 2: 1},
    "sphere": {0: 1, 1: 0, 2: 1},
}


def sample_torus(N, R=3.0, r=1.0, noise_std=0.0, seed=None):
    """Sample N points uniformly from a torus embedded in R³.

    Parameters
    ----------
    N : int
    R : float  — major radius (center of tube to center of torus)
    r : float  — minor radius (tube radius)
    noise_std : float  — isotropic Gaussian noise added to coordinates
    seed : int or None

    Returns
    -------
    pts : np.ndarray, shape (N, 3)
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, N)
    v = rng.uniform(0, 2 * np.pi, N)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    pts = np.stack([x, y, z], axis=1)
    if noise_std > 0:
        pts += rng.normal(0, noise_std, pts.shape)
    return pts


def sample_sphere(N, radius=1.0, noise_std=0.0, seed=None):
    """Sample N points uniformly from a sphere of given radius.

    Parameters
    ----------
    N : int
    radius : float
    noise_std : float
    seed : int or None

    Returns
    -------
    pts : np.ndarray, shape (N, 3)
    """
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius
    if noise_std > 0:
        pts += rng.normal(0, noise_std, pts.shape)
    return pts
