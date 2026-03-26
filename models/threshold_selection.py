"""Adaptive threshold selection for simplicial complex construction.

Given a point cloud and Gaussian kernel bandwidth sigma, sweeps threshold
values and uses the Kneedle algorithm to find the point where simplex counts
explode — the optimal threshold sits at that knee.

The threshold controls which edges exist: an edge (i,j) is included when
    exp(-d²(i,j) / sigma) >= threshold
Lower threshold → more edges/triangles.  We sweep from high to low and look
for the knee in the edge-count curve (the inflection point before explosion).
"""

import torch
import numpy as np


def _count_simplices(adj, threshold):
    """Count edges and triangles for a given adjacency matrix and threshold.

    Parameters
    ----------
    adj : torch.Tensor, shape (N, N)
        Gaussian kernel weight matrix (before thresholding).
    threshold : float
        Edges exist where adj[i,j] >= threshold.

    Returns
    -------
    n_edges : int
    n_triangles : int
    """
    n = adj.shape[0]
    # Edges: upper triangular entries above threshold
    i, j = torch.triu_indices(n, n, offset=1)
    edge_mask = adj[i, j] >= threshold
    n_edges = int(edge_mask.sum().item())

    # Triangles: all 3-vertex combinations where all 3 edge weights >= 2*threshold
    # (matches the SWT.compute_B2 logic)
    if n_edges == 0 or n > 200:
        # For large n, approximate triangle count from edges only
        # to avoid O(n³) combination enumeration
        if n > 200:
            # Sample-based estimate: check a random subset of triples
            edge_set = set()
            ei = i[edge_mask].tolist()
            ej = j[edge_mask].tolist()
            for a, b in zip(ei, ej):
                edge_set.add((a, b))

            rng = np.random.default_rng(0)
            n_triangles = 0
            n_samples = min(50000, n * (n - 1) * (n - 2) // 6)
            for _ in range(n_samples):
                triple = sorted(rng.choice(n, 3, replace=False))
                a, b, c = triple
                if (a, b) in edge_set and (b, c) in edge_set and (a, c) in edge_set:
                    if (adj[a, b] > 2 * threshold and
                        adj[b, c] > 2 * threshold and
                        adj[a, c] > 2 * threshold):
                        n_triangles += 1
            # Scale estimate
            total_triples = n * (n - 1) * (n - 2) // 6
            n_triangles = int(n_triangles * total_triples / n_samples)
        else:
            n_triangles = 0
        return n_edges, n_triangles

    combos = torch.combinations(torch.arange(n), r=3)
    ci, cj, ck = combos.T
    tri_mask = (
        (adj[ci, cj] > 2 * threshold)
        & (adj[cj, ck] > 2 * threshold)
        & (adj[ci, ck] > 2 * threshold)
    )
    n_triangles = int(tri_mask.sum().item())
    return n_edges, n_triangles


def _kneedle(x, y, direction="increasing"):
    """Find the knee point of a curve using the Kneedle algorithm.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Monotonically ordered x-values.
    y : np.ndarray, shape (n,)
        Corresponding y-values.
    direction : str
        "increasing" if y grows with x (we look for the concave knee),
        "decreasing" if y shrinks with x.

    Returns
    -------
    knee_idx : int
        Index of the knee point in the input arrays.
    """
    # Normalize to [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    if direction == "increasing":
        # Difference curve: distance from the diagonal (0,0)→(1,1)
        diff = y_norm - x_norm
    else:
        # For decreasing: distance from the anti-diagonal (0,1)→(1,0)
        diff = (1 - y_norm) - x_norm

    knee_idx = int(np.argmax(diff))
    return knee_idx


def select_threshold(point_cloud, sigma, n_steps=50, threshold_range=(0.01, 0.95)):
    """Select optimal threshold for simplicial complex construction.

    Sweeps threshold values from high (few edges) to low (many edges),
    counts edges at each value, and returns the knee point — where the
    edge count starts to explode.

    Parameters
    ----------
    point_cloud : torch.Tensor, shape (N, d)
        Input point cloud.
    sigma : float
        Gaussian kernel bandwidth.
    n_steps : int
        Number of threshold values to sweep.
    threshold_range : tuple of (float, float)
        (min_threshold, max_threshold) to sweep over.

    Returns
    -------
    optimal_threshold : float
        The selected threshold at the knee point.
    sweep_data : dict
        Dictionary with keys:
        - "thresholds": np.ndarray of swept threshold values
        - "n_edges": np.ndarray of edge counts
        - "n_triangles": np.ndarray of triangle counts
        - "knee_idx": int, index of the knee point
    """
    with torch.no_grad():
        # Compute pairwise squared distances
        G = point_cloud @ point_cloud.T
        diag = G.diag().unsqueeze(0)
        D_sq = diag + diag.T - 2 * G
        D_sq = D_sq.clamp_min(0)

        # Gaussian kernel (before thresholding)
        W = torch.exp(-D_sq / sigma)

        # Sweep thresholds from high to low
        thresholds = np.linspace(threshold_range[1], threshold_range[0], n_steps)
        edge_counts = np.zeros(n_steps, dtype=np.int64)
        tri_counts = np.zeros(n_steps, dtype=np.int64)

        for i, th in enumerate(thresholds):
            n_e, n_t = _count_simplices(W, th)
            edge_counts[i] = n_e
            tri_counts[i] = n_t

        # Kneedle on edge count curve
        # x = thresholds (decreasing), y = edge_counts (increasing)
        # We reverse so x is increasing for the algorithm
        x = thresholds[::-1].copy()  # now increasing (low → high threshold)
        y = edge_counts[::-1].copy()  # now decreasing (many → few edges)

        knee_idx_reversed = _kneedle(x, y, direction="decreasing")
        # Map back to original indexing
        knee_idx = n_steps - 1 - knee_idx_reversed

        optimal_threshold = float(thresholds[knee_idx])

    return optimal_threshold, {
        "thresholds": thresholds,
        "n_edges": edge_counts,
        "n_triangles": tri_counts,
        "knee_idx": knee_idx,
    }
