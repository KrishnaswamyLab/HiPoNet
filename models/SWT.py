import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.hodge_laplacian import (
    cayley_menger_volumes,
    geometric_hodge_laplacian_torch,
)


class PoolingAttention(nn.Module):
    """DeepSet attention pooling over a set of simplices.

    Computes a learned attention weight per simplex, then returns the
    weighted sum.  Attention weights are stored in ``self.last_weights``
    after each forward call for downstream interpretability.
    """

    def __init__(self, feat_dim, hidden_dim=32):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.last_weights = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (n_elements, feat_dim)

        Returns
        -------
        pooled : torch.Tensor, shape (feat_dim,)
        """
        scores = self.score_fn(x).squeeze(-1)  # (n_elements,)
        weights = F.softmax(scores, dim=0)  # (n_elements,)
        self.last_weights = weights.detach()
        return (weights.unsqueeze(-1) * x).sum(0)  # (feat_dim,)


class SimplicialWaveletTransform:
    """Simplicial wavelet transform with optional geometric Hodge Laplacian.

    When use_geometric_laplacian=False (default), uses the original
    combinatorial transition matrices derived from weighted boundary matrices.

    When use_geometric_laplacian=True, builds a proper oriented simplicial
    complex (optionally in Diffusion Maps coordinates), computes mass matrices
    from simplex volumes, and assembles the metric-aware Hodge Laplacian.
    Transition matrices for message passing are then derived from this
    geometric Laplacian.

    Parameters
    ----------
    adj : torch.Tensor
        Adjacency/weight matrix, shape (N, N).
    ro : torch.Tensor
        Node features, shape (N, d).
    threshold : float
        Weight threshold for simplex inclusion.
    device : torch.device or str
    use_geometric_laplacian : bool
        If True, use geometric Hodge Laplacian with mass matrices derived from
        diffusion distances (no eigendecomposition, fully differentiable).
    sq_diff_dists : torch.Tensor or None, shape (N, N)
        Pairwise squared diffusion distances. Required when
        use_geometric_laplacian is True.
    """

    def __init__(
        self,
        adj,
        ro,
        threshold,
        device,
        use_geometric_laplacian=False,
        sq_diff_dists=None,
    ):
        self.adj = adj
        self.device = device
        self.X = ro
        self.indices = []
        self.threshold = threshold
        self.use_geometric_laplacian = use_geometric_laplacian

        # Build boundary matrices and simplex features via original method
        self.B = [None, self.compute_B1(), self.compute_B2()]
        X = self.calculate_simplex_features()
        self.X = [ro, X[0], X[1]]
        del X

        if use_geometric_laplacian:
            self._build_geometric_transition_matrices(sq_diff_dists)
        else:
            self.P_B, self.P_L, self.P_U = self._get_transition_matrix()

    def compute_B1(self):
        n = self.adj.shape[0]
        i, j = torch.triu_indices(n, n, offset=1).to(self.device)
        weights = self.adj[i, j]

        non_zero_mask = weights > 0
        i, j = i[non_zero_mask], j[non_zero_mask]
        weights = weights[non_zero_mask]
        num_edges = len(weights)

        self.edges = torch.stack((i, j)).T

        B1 = torch.zeros((n, num_edges), device=self.adj.device)
        B1[i, torch.arange(num_edges)] = weights
        B1[j, torch.arange(num_edges)] = weights
        return B1

    def compute_B2(self):
        n = self.adj.shape[0]
        i, j = self.edges.T
        edge_weights = self.adj[i, j]
        num_edges = len(edge_weights)

        connected = self.adj >= self.threshold
        common_neighbors = connected[i] & connected[j]
        common_neighbors &= torch.arange(n, device=self.adj.device)[None, :] > j[:, None]
        edge_rows, third_vertices = torch.where(common_neighbors)
        self.triangles = torch.stack(
            [i[edge_rows], j[edge_rows], third_vertices], dim=1
        )
        if len(self.triangles) > 250:
            selected = torch.randint(0, len(self.triangles), (250,), device=self.adj.device)
            self.triangles = self.triangles[selected]
        num_triangles = self.triangles.shape[0]
        B2 = torch.zeros((num_edges, num_triangles), device=self.adj.device)
        if num_triangles:
            edge_lookup = torch.full((n, n), -1, dtype=torch.long, device=self.adj.device)
            edge_ids = torch.arange(num_edges, device=self.adj.device)
            edge_lookup[i, j] = edge_ids
            edge_lookup[j, i] = edge_ids
            triangle_edges = torch.stack([
                edge_lookup[self.triangles[:, 0], self.triangles[:, 1]],
                edge_lookup[self.triangles[:, 0], self.triangles[:, 2]],
                edge_lookup[self.triangles[:, 1], self.triangles[:, 2]],
            ], dim=1)
            columns = torch.arange(num_triangles, device=self.adj.device).repeat_interleave(3)
            rows = triangle_edges.reshape(-1)
            B2[rows, columns] = edge_weights[rows]
        return B2

    def calculate_simplex_features(self):
        X1 = self.X[self.edges].mean(1)
        X2 = self.X[self.triangles].mean(1)
        return [X1, X2]

    def _get_laplacians(self):
        lower_laplacians = [None] * len(self.B)
        upper_laplacians = [None] * len(self.B)
        for i in range(1, len(self.B)):
            lower_laplacians[i] = self.B[i].T @ self.B[i]
        for i in range(0, len(self.B) - 1):
            upper_laplacians[i] = self.B[i + 1] @ self.B[i + 1].T
        return lower_laplacians, upper_laplacians

    def _get_transition_matrix(self):
        P_B = [None] * len(self.B)
        P_U = [None] * len(self.B)
        P_L = [None] * len(self.B)
        for i in range(len(self.B)):
            if self.B[i] is not None:
                inverse_degree = (self.B[i].sum(axis=1) + 1.0).reciprocal()
                P_B[i] = inverse_degree[:, None] * self.B[i]
        for i in range(1, len(self.B)):
            ul = self.B[i].T @ self.B[i]
            inverse_degree = (ul.sum(axis=1) + 1.0).reciprocal()
            P_L[i] = ul * inverse_degree[None, :]
        for i in range(0, len(self.B) - 1):
            ll = self.B[i + 1] @ self.B[i + 1].T
            inverse_degree = (ll.sum(axis=1) + 1.0).reciprocal()
            P_U[i] = ll * inverse_degree[None, :]
        return P_B, P_L, P_U

    def _build_geometric_transition_matrices(self, sq_diff_dists):
        """Build transition matrices from the geometric Hodge Laplacian.

        All computation stays in torch — no numpy, no detach — so gradients
        flow back through sq_diff_dists to the learnable alphas.

        Parameters
        ----------
        sq_diff_dists : torch.Tensor, shape (N, N)
            Pairwise squared diffusion distances, differentiable w.r.t. alpha.
        """
        n_vertices = self.adj.shape[0]

        # 0-simplex volumes: uniform weights
        v0 = sq_diff_dists.new_full((n_vertices,), 1.0 / max(n_vertices, 1))

        # 1-simplex volumes: edge lengths via Cayley-Menger (reduces to dist)
        if self.edges.shape[0] > 0:
            v1 = cayley_menger_volumes(self.edges, sq_diff_dists)
        else:
            v1 = sq_diff_dists.new_ones(self.B[1].shape[1])

        # 2-simplex volumes: triangle areas via Cayley-Menger
        if isinstance(self.triangles, np.ndarray):
            tris = torch.from_numpy(self.triangles.astype(np.int64)).to(self.device)
        else:
            tris = self.triangles.long().to(self.device)

        if tris.shape[0] > 0:
            v2 = cayley_menger_volumes(tris, sq_diff_dists)
        else:
            v2 = sq_diff_dists.new_ones(self.B[2].shape[1])

        # Boundary matrices stay as torch tensors (with grad)
        B1 = self.B[1].float()
        B2 = self.B[2].float()

        # Assemble geometric Hodge Laplacians — pure torch, differentiable
        delta_0 = geometric_hodge_laplacian_torch(
            B_k=None, B_kp1=B1, v_km1=None, v_k=v0, v_kp1=v1
        )
        delta_1 = geometric_hodge_laplacian_torch(
            B_k=B1, B_kp1=B2, v_km1=v0, v_k=v1, v_kp1=v2
        )
        delta_2 = (
            geometric_hodge_laplacian_torch(
                B_k=B2, B_kp1=None, v_km1=v1, v_k=v2, v_kp1=None
            )
            if B2.shape[1] > 0
            else None
        )

        deltas = [delta_0, delta_1, delta_2]

        def _normalize(delta):
            d_sum = delta.sum(dim=1)
            return delta * (d_sum + 1.0).reciprocal()[None, :]

        # Boundary transition matrices (from self.B which carries alpha grad)
        P_B = [None] * len(self.B)
        for i in range(len(self.B)):
            if self.B[i] is not None:
                inverse_degree = (self.B[i].sum(axis=1) + 1.0).reciprocal()
                P_B[i] = inverse_degree[:, None] * self.B[i]

        P_U = [None] * len(self.B)
        P_L = [None] * len(self.B)

        for k in range(len(self.B) - 1):
            if deltas[k] is not None:
                P_U[k] = _normalize(deltas[k])

        for k in range(1, len(self.B)):
            if deltas[k] is not None:
                P_L[k] = _normalize(deltas[k])

        self.P_B = P_B
        self.P_L = P_L
        self.P_U = P_U

    def message_passing(self, X, include_boundary):
        neighbors = []
        aggregate = []
        for k in range(len(X)):
            X_l = torch.zeros(X[k].shape).to(self.device)
            X_u = torch.zeros(X[k].shape).to(self.device)
            X_b = torch.zeros(X[k].shape).to(self.device)
            X_c = torch.zeros(X[k].shape).to(self.device)
            if self.P_U[k] is not None:
                X_u = self.P_U[k] @ X[k]
            if self.P_L[k] is not None:
                X_l = self.P_L[k] @ X[k]
            if include_boundary:
                if k < len(X) - 1:
                    if self.B[k] is not None:
                        X_b = self.P_B[k + 1] @ X[k + 1]
                if self.B[k] is not None:
                    X_c = self.P_B[k].T @ X[k - 1]
            neighbors.append([X_b, X_c, X_l, X_u])
            aggregate.append(X[k] / 5 + X_b + X_c + X_l + X_u)
        return neighbors, aggregate

    def calculate_Z(self, J, include_boundary):
        Z_agg = []
        Z_neigh = []
        for i in range(J):
            if i == 0:
                neigh, agg = self.message_passing(self.X, include_boundary)
                Z_agg.append(agg)
                Z_neigh.append(neigh)
            else:
                neigh, agg = self.message_passing(Z_agg[-1], include_boundary)
                Z_agg.append(agg)
                Z_neigh.append(neigh)
        return Z_agg, Z_neigh

    def scattering(self, X, Z_neigh, index, J):
        psi = []
        for i in index:
            p = []
            for j in range(J + 1):
                out = []
                if j == 0:
                    for k in range(len(X)):
                        out.append(torch.zeros_like(X[k]))
                elif j == J:
                    for k in range(len(X)):
                        out.append(torch.zeros_like(X[k]))
                else:
                    for k in range(len(X)):
                        out.append(
                            torch.abs(
                                Z_neigh[j - 1][k][index[i]]
                                - Z_neigh[j][k][index[i]]
                            )
                        )
                p.append(out)
            psi.append(p)
        return psi

    def agg(self, psi, X, index, J):
        Psi = []
        for j in range(J):
            psi_j = []
            for k in range(len(X)):
                psi_j.append(
                    (
                        X[k]
                        + psi[index["B"]][j][k]
                        + psi[index["C"]][j][k]
                        + psi[index["L"]][j][k]
                        + psi[index["U"]][j][k]
                    )
                    / 5
                )
            Psi.append(psi_j)
        return Psi

    def calculate_wavelet_coeff(self, J, output_size=None, include_boundary=True, pool_attn=None):
        index = {"B": 0, "C": 1, "L": 2, "U": 3}
        Z_agg, Z_neigh = self.calculate_Z(J, include_boundary)
        psi = self.scattering(self.X, Z_neigh, index, J)
        Psi_j = self.agg(psi, self.X, index, J)
        PSI = []
        for PsiX in Psi_j:
            parts = []
            for dim_idx, k in enumerate(PsiX):
                if pool_attn is not None:
                    parts.append(pool_attn[dim_idx](k))
                else:
                    parts.append(k.sum(0))
            PSI.append(torch.cat(parts, dim=0))
        return torch.cat(PSI)
