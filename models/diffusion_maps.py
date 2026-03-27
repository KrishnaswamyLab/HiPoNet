import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


def diffusion_maps(
    X,
    n_components=10,
    epsilon=None,
    alpha=1.0,
    diffusion_time=1.0,
    n_neighbors=None,
):
    """Compute Diffusion Maps embedding of a point cloud.

    Maps X from (N, D) ambient space to (N, m) coordinates where Euclidean
    distances approximate geodesic distances on the underlying manifold.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Input point cloud. Accepts numpy arrays or torch tensors.
    n_components : int
        Number of diffusion map coordinates to keep.
    epsilon : float or None
        Gaussian kernel bandwidth. If None, auto-tuned as the median of
        squared pairwise distances.
    alpha : float
        Density normalization exponent. alpha=1.0 gives density-invariant
        (Laplace-Beltrami) behavior. alpha=0.0 gives the standard
        normalized graph Laplacian.
    diffusion_time : float
        Diffusion time parameter t. Coordinates are scaled by lambda^(t/2).
    n_neighbors : int or None
        If set, sparsify the kernel using k-nearest neighbors. Recommended
        for large N to reduce memory and computation.

    Returns
    -------
    Y : np.ndarray, shape (N, n_components)
        Diffusion map coordinates.
    eigenvalues : np.ndarray, shape (n_components,)
        The eigenvalues (excluding the trivial eigenvalue=1).
    """
    if hasattr(X, "numpy"):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]

    # Pairwise squared distances
    sq_dists = squareform(pdist(X, metric="sqeuclidean"))

    # Auto-tune epsilon
    if epsilon is None:
        epsilon = np.median(sq_dists[sq_dists > 0])

    # Gaussian kernel: W_ij = exp(-||x_i - x_j||^2 / (4 * epsilon))
    W = np.exp(-sq_dists / (4.0 * epsilon))

    # k-NN sparsification
    if n_neighbors is not None and n_neighbors < N:
        W_sparse = np.zeros_like(W)
        for i in range(N):
            # Keep only k nearest neighbors (plus self)
            idx = np.argsort(sq_dists[i])[:n_neighbors + 1]
            W_sparse[i, idx] = W[i, idx]
        # Symmetrize
        W = np.maximum(W_sparse, W_sparse.T)

    # Alpha-normalization for density invariance
    q = W.sum(axis=1)
    q_alpha = np.power(q, alpha)
    # W_ij <- W_ij / (q_i^alpha * q_j^alpha)
    W = W / np.outer(q_alpha, q_alpha)

    # Row-normalize to get Markov matrix P
    d = W.sum(axis=1)
    d_inv = 1.0 / np.maximum(d, 1e-15)

    # Form the symmetric conjugate for eigsh: D^{-1/2} W D^{-1/2}
    # This has the same eigenvalues as D^{-1} W but is symmetric
    d_inv_sqrt = np.sqrt(d_inv)
    W_sym = W * np.outer(d_inv_sqrt, d_inv_sqrt)
    W_sym = 0.5 * (W_sym + W_sym.T)  # enforce symmetry

    W_sym_sparse = csr_matrix(W_sym)

    # Compute top eigenvalues/eigenvectors (including trivial)
    n_eigs = min(n_components + 1, N - 1)
    eigenvalues, eigenvectors = eigsh(W_sym_sparse, k=n_eigs, which="LM")

    # Sort descending
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Convert back to right eigenvectors of P: phi = D^{-1/2} psi
    eigenvectors = eigenvectors * d_inv_sqrt[:, None]

    # Drop trivial eigenvector (constant, eigenvalue~1)
    eigenvalues = eigenvalues[1 : n_components + 1]
    eigenvectors = eigenvectors[:, 1 : n_components + 1]

    # DM embedding: y_i = (lambda_k^{t/2} * phi_k(i))
    scaling = np.power(np.maximum(eigenvalues, 0.0), diffusion_time / 2.0)
    Y = eigenvectors * scaling[None, :]

    return Y, eigenvalues
