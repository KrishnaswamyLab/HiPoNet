import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.hodge_laplacian import (
    cayley_menger_volumes,
    geometric_hodge_laplacian_torch,
)


class FactorizedTransition:
    """Row-normalized linear operator evaluated through sparse incidence factors."""

    def __init__(self, size, apply_raw, reference):
        self.shape = (size, size)
        self._apply_raw = apply_raw
        ones = reference.new_ones((size, 1))
        self._inverse_degree = (apply_raw(ones).squeeze(1) + 1.0).reciprocal()

    def matmul(self, features):
        return self._inverse_degree[:, None] * self._apply_raw(features)


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
        max_triangles=250,
        sparse_operators=False,
        max_neighbors=None,
    ):
        self.adj = adj
        self.device = device
        self.X = ro
        self.indices = []
        self.threshold = threshold
        self.use_geometric_laplacian = use_geometric_laplacian
        self.max_triangles = max_triangles
        self.sparse_operators = sparse_operators
        self.max_neighbors = max_neighbors

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

        edge_ids = torch.arange(num_edges, device=self.adj.device)
        if self.sparse_operators:
            indices = torch.stack(
                [torch.cat([i, j]), torch.cat([edge_ids, edge_ids])]
            )
            values = torch.cat([weights, weights])
            B1 = torch.sparse_coo_tensor(
                indices, values, (n, num_edges), device=self.adj.device
            ).coalesce()
        else:
            B1 = torch.zeros((n, num_edges), device=self.adj.device)
            B1[i, edge_ids] = weights
            B1[j, edge_ids] = weights
        return B1

    def _enumerate_sparse_triangles(self):
        """Enumerate triangles from a degree-bounded adjacency without E x N storage."""
        n = self.adj.shape[0]
        if n < 3 or self.max_neighbors is None or self.max_neighbors < 2:
            return torch.empty((0, 3), dtype=torch.long, device=self.adj.device)

        connected = self.adj > 0
        connected.fill_diagonal_(False)
        k = min(self.max_neighbors, n - 1)
        scores = self.adj.masked_fill(~connected, float("-inf"))
        values, neighbors = torch.topk(scores, k=k, dim=1)
        valid_neighbors = torch.isfinite(values)

        pair_positions = torch.combinations(
            torch.arange(k, device=self.adj.device), r=2
        )
        if pair_positions.numel() == 0:
            return torch.empty((0, 3), dtype=torch.long, device=self.adj.device)

        left = pair_positions[:, 0]
        right = pair_positions[:, 1]
        j = neighbors[:, left]
        k_vertex = neighbors[:, right]
        j, k_vertex = torch.minimum(j, k_vertex), torch.maximum(j, k_vertex)
        i = torch.arange(n, device=self.adj.device)[:, None].expand_as(j)
        valid = (
            valid_neighbors[:, left]
            & valid_neighbors[:, right]
            & (i < j)
            & (j < k_vertex)
            & connected[j, k_vertex]
        )
        return torch.stack([i[valid], j[valid], k_vertex[valid]], dim=1)

    def compute_B2(self):
        n = self.adj.shape[0]
        i, j = self.edges.T
        edge_weights = self.adj[i, j]
        num_edges = len(edge_weights)

        if self.sparse_operators:
            self.triangles = self._enumerate_sparse_triangles()
        else:
            connected = self.adj >= self.threshold
            common_neighbors = connected[i] & connected[j]
            common_neighbors &= torch.arange(n, device=self.adj.device)[None, :] > j[:, None]
            edge_rows, third_vertices = torch.where(common_neighbors)
            self.triangles = torch.stack(
                [i[edge_rows], j[edge_rows], third_vertices], dim=1
            )
        if self.max_triangles is not None and len(self.triangles) > self.max_triangles:
            selected = torch.randperm(
                len(self.triangles), device=self.adj.device
            )[: self.max_triangles]
            self.triangles = self.triangles[selected]
        num_triangles = self.triangles.shape[0]
        if self.sparse_operators:
            B2 = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.adj.device),
                torch.empty(0, dtype=self.adj.dtype, device=self.adj.device),
                (num_edges, num_triangles),
                device=self.adj.device,
            ).coalesce()
        else:
            B2 = torch.zeros((num_edges, num_triangles), device=self.adj.device)
        if num_triangles:
            edge_keys = i * n + j
            triangle_keys = torch.stack(
                [
                    self.triangles[:, 0] * n + self.triangles[:, 1],
                    self.triangles[:, 0] * n + self.triangles[:, 2],
                    self.triangles[:, 1] * n + self.triangles[:, 2],
                ],
                dim=1,
            )
            triangle_edges = torch.searchsorted(edge_keys, triangle_keys)
            columns = torch.arange(num_triangles, device=self.adj.device).repeat_interleave(3)
            rows = triangle_edges.reshape(-1)
            values = edge_weights[rows]
            if self.sparse_operators:
                B2 = torch.sparse_coo_tensor(
                    torch.stack([rows, columns]),
                    values,
                    (num_edges, num_triangles),
                    device=self.adj.device,
                ).coalesce()
            else:
                B2[rows, columns] = values
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
                P_B[i] = self._normalize_rows(self.B[i])
        for i in range(1, len(self.B)):
            if self.B[i].is_sparse:
                boundary = self.B[i]
                P_L[i] = FactorizedTransition(
                    boundary.shape[1],
                    lambda x, boundary=boundary: self._apply_operator(
                        boundary.T, self._apply_operator(boundary, x)
                    ),
                    self.X[i],
                )
            else:
                ul = self.B[i].T @ self.B[i]
                P_L[i] = self._normalize_rows(ul)
        for i in range(0, len(self.B) - 1):
            if self.B[i + 1].is_sparse:
                boundary = self.B[i + 1]
                P_U[i] = FactorizedTransition(
                    boundary.shape[0],
                    lambda x, boundary=boundary: self._apply_operator(
                        boundary, self._apply_operator(boundary.T, x)
                    ),
                    self.X[i],
                )
            else:
                ll = self.B[i + 1] @ self.B[i + 1].T
                P_U[i] = self._normalize_rows(ll)
        return P_B, P_L, P_U

    @staticmethod
    def _scale_sparse(op, row_scale=None, col_scale=None):
        op = op.coalesce()
        indices = op.indices()
        values = op.values()
        if row_scale is not None:
            values = values * row_scale[indices[0]]
        if col_scale is not None:
            values = values * col_scale[indices[1]]
        return torch.sparse_coo_tensor(
            indices, values, op.shape, device=op.device
        ).coalesce()

    @classmethod
    def _normalize_rows(cls, op):
        if op.is_sparse:
            row_sum = torch.sparse.sum(op, dim=1).to_dense()
            return cls._scale_sparse(op, row_scale=(row_sum + 1.0).reciprocal())
        inverse_degree = (op.sum(dim=1) + 1.0).reciprocal()
        return inverse_degree[:, None] * op

    @staticmethod
    def _apply_operator(op, features):
        if isinstance(op, FactorizedTransition):
            return op.matmul(features)
        if op.is_sparse:
            return torch.sparse.mm(op, features)
        return op @ features

    def _geometric_upper_transition(self, boundary, simplex_mass, coface_mass, reference):
        if boundary is None or boundary.shape[1] == 0:
            return None
        inverse_mass = simplex_mass.reciprocal()

        def apply_raw(features):
            lifted = self._apply_operator(boundary.T, features)
            return inverse_mass[:, None] * self._apply_operator(
                boundary, coface_mass[:, None] * lifted
            )

        return FactorizedTransition(boundary.shape[0], apply_raw, reference)

    def _geometric_lower_transition(self, boundary, face_mass, simplex_mass, reference):
        if boundary is None or boundary.shape[1] == 0:
            return None
        inverse_face_mass = face_mass.reciprocal()

        def apply_raw(features):
            projected = self._apply_operator(
                boundary, simplex_mass[:, None] * features
            )
            return self._apply_operator(
                boundary.T, inverse_face_mass[:, None] * projected
            )

        return FactorizedTransition(boundary.shape[1], apply_raw, reference)

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

        if not self.sparse_operators:
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

        # Boundary transition matrices (from self.B which carries alpha grad)
        P_B = [None] * len(self.B)
        for i in range(len(self.B)):
            if self.B[i] is not None:
                P_B[i] = self._normalize_rows(self.B[i])

        if self.sparse_operators:
            P_U = [
                self._geometric_upper_transition(B1, v0, v1, self.X[0]),
                self._geometric_upper_transition(B2, v1, v2, self.X[1]),
                None,
            ]
            P_L = [
                None,
                self._geometric_lower_transition(B1, v0, v1, self.X[1]),
                self._geometric_lower_transition(B2, v1, v2, self.X[2]),
            ]
        else:
            deltas = [delta_0, delta_1, delta_2]
            P_U = [None] * len(self.B)
            P_L = [None] * len(self.B)
            for k in range(len(self.B) - 1):
                if deltas[k] is not None:
                    P_U[k] = self._normalize_rows(deltas[k])
            for k in range(1, len(self.B)):
                if deltas[k] is not None:
                    P_L[k] = self._normalize_rows(deltas[k])

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
                X_u = self._apply_operator(self.P_U[k], X[k])
            if self.P_L[k] is not None:
                X_l = self._apply_operator(self.P_L[k], X[k])
            if include_boundary:
                if k < len(X) - 1:
                    if self.B[k] is not None:
                        X_b = self._apply_operator(self.P_B[k + 1], X[k + 1])
                if self.B[k] is not None:
                    X_c = self._apply_operator(self.P_B[k].T, X[k - 1])
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
                elif self.sparse_operators:
                    parts.append(k.mean(0) if k.shape[0] else k.new_zeros(k.shape[1]))
                else:
                    parts.append(k.sum(0))
            PSI.append(torch.cat(parts, dim=0))
        return torch.cat(PSI)
