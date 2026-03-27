import numpy as np
import torch
from math import factorial
from scipy.sparse import diags, eye, csr_matrix


def compute_simplex_volumes(sc, points):
    """Compute the k-dimensional volume of every simplex in the complex.

    Parameters
    ----------
    sc : SimplicialComplex
        The simplicial complex (provides simplex lists).
    points : array-like, shape (N, m)
        Vertex coordinates (typically Diffusion Maps embedding).

    Returns
    -------
    volumes : dict
        Mapping dimension k -> np.ndarray of shape (n_k,) with the volume
        of each k-simplex. Volumes are clamped to a positive floor.
    """
    if hasattr(points, "numpy"):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)

    volumes = {}
    floor = 1e-10

    # 0-simplices: uniform weight 1/N
    n0 = sc.num_simplices(0)
    if n0 > 0:
        volumes[0] = np.full(n0, 1.0 / n0)

    # 1-simplices (edges): Euclidean length
    if sc.num_simplices(1) > 0:
        vols = np.empty(sc.num_simplices(1))
        for idx, (i, j) in enumerate(sc.simplices[1]):
            vols[idx] = np.linalg.norm(points[i] - points[j])
        vols = np.maximum(vols, floor)
        volumes[1] = vols

    # k-simplices for k >= 2: (1/k!) * sqrt(det(Gram matrix))
    for k in range(2, sc.max_dim + 1):
        if sc.num_simplices(k) == 0:
            continue
        vols = np.empty(sc.num_simplices(k))
        for idx, simplex in enumerate(sc.simplices[k]):
            # Edge vectors from first vertex
            v0 = points[simplex[0]]
            edge_vecs = np.array([points[v] - v0 for v in simplex[1:]])
            # Gram matrix: G_ij = <e_i, e_j>
            G = edge_vecs @ edge_vecs.T
            det_G = np.linalg.det(G)
            det_G = max(det_G, 0.0)  # numerical safety
            vols[idx] = np.sqrt(det_G) / factorial(k)
        vols = np.maximum(vols, floor)
        volumes[k] = vols

    return volumes


def mass_matrix(volumes):
    """Build diagonal mass matrix M_k = diag(volumes) and its inverse.

    Parameters
    ----------
    volumes : np.ndarray, shape (n_k,)
        Volumes of k-simplices.

    Returns
    -------
    M : scipy.sparse.dia_matrix
        Diagonal mass matrix.
    M_inv : scipy.sparse.dia_matrix
        Inverse diagonal mass matrix.
    """
    M = diags(volumes)
    M_inv = diags(1.0 / volumes)
    return M, M_inv


def geometric_hodge_laplacian(B_k, B_kp1, M_km1, M_k, M_kp1):
    """Assemble the metric-aware Hodge Laplacian for degree k.

    Δ_k^(M) = M_k^{-1} B_{k+1} M_{k+1} B_{k+1}^T  +  B_k^T M_{k-1}^{-1} B_k M_k

    When all mass matrices are None or identity, this reduces to the
    combinatorial Hodge Laplacian.

    Parameters
    ----------
    B_k : sparse matrix or None
        Boundary matrix from (k-1)-simplices to k-simplices. None for k=0.
    B_kp1 : sparse matrix or None
        Boundary matrix from k-simplices to (k+1)-simplices. None for top dim.
    M_km1 : (M, M_inv) tuple or None
        Mass matrix pair for (k-1)-simplices.
    M_k : (M, M_inv) tuple or None
        Mass matrix pair for k-simplices.
    M_kp1 : (M, M_inv) tuple or None
        Mass matrix pair for (k+1)-simplices.

    Returns
    -------
    delta_k : np.ndarray
        The geometric Hodge Laplacian matrix (dense).
    """
    # Determine size from whatever boundary matrix is available
    if B_kp1 is not None:
        n = B_kp1.shape[0]
    elif B_k is not None:
        n = B_k.shape[1]
    else:
        raise ValueError("At least one of B_k or B_kp1 must be provided")

    delta_k = np.zeros((n, n))

    # Upper part: M_k^{-1} @ B_{k+1} @ M_{k+1} @ B_{k+1}^T
    if B_kp1 is not None:
        B_kp1_dense = B_kp1.toarray() if hasattr(B_kp1, "toarray") else np.asarray(B_kp1)
        if M_k is not None and M_kp1 is not None:
            _, M_k_inv = M_k
            M_kp1_mat, _ = M_kp1
            M_k_inv_dense = M_k_inv.toarray() if hasattr(M_k_inv, "toarray") else np.asarray(M_k_inv)
            M_kp1_dense = M_kp1_mat.toarray() if hasattr(M_kp1_mat, "toarray") else np.asarray(M_kp1_mat)
            # Compute right-to-left: B_{k+1}^T first, then M_{k+1}, etc.
            upper = M_k_inv_dense @ B_kp1_dense @ M_kp1_dense @ B_kp1_dense.T
        else:
            # Combinatorial: just B_{k+1} @ B_{k+1}^T
            upper = B_kp1_dense @ B_kp1_dense.T
        delta_k += upper

    # Lower part: B_k^T @ M_{k-1}^{-1} @ B_k @ M_k
    if B_k is not None:
        B_k_dense = B_k.toarray() if hasattr(B_k, "toarray") else np.asarray(B_k)
        if M_km1 is not None and M_k is not None:
            _, M_km1_inv = M_km1
            M_k_mat, _ = M_k
            M_km1_inv_dense = M_km1_inv.toarray() if hasattr(M_km1_inv, "toarray") else np.asarray(M_km1_inv)
            M_k_dense = M_k_mat.toarray() if hasattr(M_k_mat, "toarray") else np.asarray(M_k_mat)
            lower = B_k_dense.T @ M_km1_inv_dense @ B_k_dense @ M_k_dense
        else:
            # Combinatorial: just B_k^T @ B_k
            lower = B_k_dense.T @ B_k_dense
        delta_k += lower

    return delta_k


def combinatorial_hodge_laplacian(B_k, B_kp1):
    """Combinatorial Hodge Laplacian (M_k = I for all k).

    Δ_k^comb = B_{k+1} @ B_{k+1}^T + B_k^T @ B_k

    Parameters
    ----------
    B_k : sparse matrix or None
        Boundary matrix from (k-1) to k.
    B_kp1 : sparse matrix or None
        Boundary matrix from k to (k+1).

    Returns
    -------
    delta_k : np.ndarray
        The combinatorial Hodge Laplacian.
    """
    return geometric_hodge_laplacian(B_k, B_kp1, None, None, None)


def symmetric_hodge_laplacian(delta_k, M_k):
    """Compute the symmetric conjugate of the geometric Hodge Laplacian.

    Δ̃_k = M_k^{1/2} @ Δ_k^(M) @ M_k^{-1/2}

    This matrix is symmetric in the standard Euclidean sense and has the
    same eigenvalues as Δ_k^(M).

    Parameters
    ----------
    delta_k : np.ndarray
        The geometric Hodge Laplacian.
    M_k : (M, M_inv) tuple
        Mass matrix pair for k-simplices.

    Returns
    -------
    delta_tilde : np.ndarray
        The symmetric conjugate Hodge Laplacian.
    """
    M_mat, _ = M_k
    M_dense = M_mat.toarray() if hasattr(M_mat, "toarray") else np.asarray(M_mat)
    diag_vals = np.diag(M_dense)
    sqrt_diag = np.sqrt(np.maximum(diag_vals, 1e-15))
    inv_sqrt_diag = 1.0 / sqrt_diag

    # Δ̃ = diag(sqrt) @ Δ @ diag(inv_sqrt)
    delta_tilde = (sqrt_diag[:, None] * delta_k) * inv_sqrt_diag[None, :]

    # Enforce exact symmetry
    delta_tilde = 0.5 * (delta_tilde + delta_tilde.T)
    return delta_tilde


def hodge_laplacian_to_torch(delta_k, device="cpu"):
    """Convert a Hodge Laplacian from numpy to a torch tensor.

    Parameters
    ----------
    delta_k : np.ndarray
        Hodge Laplacian matrix.
    device : str or torch.device

    Returns
    -------
    torch.Tensor
    """
    return torch.tensor(delta_k, dtype=torch.float32, device=device)


def cayley_menger_volumes(simplex_list, sq_dist_matrix):
    """Compute simplex volumes via the Cayley-Menger determinant.

    Fully differentiable w.r.t. sq_dist_matrix (uses torch.linalg.det).

    For an n-simplex with n+1 vertices and pairwise squared distances d²_ij:
        Vol_n² = (-1)^(n+1) / (2^n · (n!)²) · det(CM)
    where CM is the (n+2)×(n+2) bordered distance matrix.

    Parameters
    ----------
    simplex_list : list of tuples or torch.Tensor, shape (S, k+1)
        Vertex indices. k+1 vertices per k-simplex.
    sq_dist_matrix : torch.Tensor, shape (N, N)
        Pairwise squared distances.

    Returns
    -------
    volumes : torch.Tensor, shape (S,)
    """
    if not isinstance(simplex_list, torch.Tensor):
        simplex_list = torch.tensor(simplex_list, dtype=torch.long,
                                    device=sq_dist_matrix.device)
    else:
        simplex_list = simplex_list.long().to(sq_dist_matrix.device)

    if simplex_list.shape[0] == 0:
        return sq_dist_matrix.new_zeros(0)

    S = simplex_list.shape[0]
    n_verts = simplex_list.shape[1]   # k+1 for a k-simplex
    n_dim = n_verts - 1               # simplex dimension k

    # Gather pairwise sq-distances for all simplices: (S, n_verts, n_verts)
    v_i = simplex_list.unsqueeze(2).expand(-1, -1, n_verts)
    v_j = simplex_list.unsqueeze(1).expand(-1, n_verts, -1)
    d_block = sq_dist_matrix[v_i, v_j]

    # Cayley-Menger matrix: (S, n_verts+1, n_verts+1)
    cm = sq_dist_matrix.new_zeros(S, n_verts + 1, n_verts + 1)
    cm[:, 0, 1:] = 1.0
    cm[:, 1:, 0] = 1.0
    cm[:, 1:, 1:] = d_block

    det = torch.linalg.det(cm)

    sign = (-1) ** (n_dim + 1)
    denom = (2 ** n_dim) * (factorial(n_dim) ** 2)
    vol_sq = sign * det / denom

    # clamp before sqrt to avoid infinite gradient at zero
    return torch.sqrt(vol_sq.clamp(min=1e-20)).clamp(min=1e-10)


def geometric_hodge_laplacian_torch(B_k, B_kp1, v_km1, v_k, v_kp1):
    """Assemble the metric-aware Hodge Laplacian in torch (fully differentiable).

    Δ_k^(M) = M_k⁻¹ B_{k+1} M_{k+1} B_{k+1}ᵀ  +  B_kᵀ M_{k-1}⁻¹ B_k M_k

    where M_k = diag(v_k). Pass None for any missing boundary or volume term
    to fall back to the combinatorial (identity-mass) contribution.

    Parameters
    ----------
    B_k   : torch.Tensor or None, shape (n_{k-1}, n_k)
    B_kp1 : torch.Tensor or None, shape (n_k, n_{k+1})
    v_km1 : torch.Tensor or None, shape (n_{k-1},)
    v_k   : torch.Tensor or None, shape (n_k,)
    v_kp1 : torch.Tensor or None, shape (n_{k+1},)

    Returns
    -------
    delta_k : torch.Tensor, shape (n_k, n_k)
    """
    ref = B_kp1 if B_kp1 is not None else B_k
    n = ref.shape[0] if B_kp1 is not None else ref.shape[1]
    delta_k = ref.new_zeros(n, n)

    # Upper part: M_k⁻¹ @ B_{k+1} @ diag(v_{k+1}) @ B_{k+1}ᵀ
    if B_kp1 is not None:
        if v_k is not None and v_kp1 is not None:
            upper = (1.0 / v_k)[:, None] * ((B_kp1 * v_kp1[None, :]) @ B_kp1.T)
        else:
            upper = B_kp1 @ B_kp1.T
        delta_k = delta_k + upper

    # Lower part: B_kᵀ @ diag(1/v_{k-1}) @ B_k @ diag(v_k)
    if B_k is not None:
        if v_km1 is not None and v_k is not None:
            lower = B_k.T @ ((1.0 / v_km1)[:, None] * B_k) * v_k[None, :]
        else:
            lower = B_k.T @ B_k
        delta_k = delta_k + lower

    return delta_k
