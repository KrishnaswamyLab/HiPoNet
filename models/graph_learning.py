import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GWT import GraphWaveletTransform
import gc

gc.enable()


def compute_dist(X):
    G = torch.matmul(X, X.T)
    D = (
        torch.reshape(torch.diag(G), (1, -1))
        + torch.reshape(torch.diag(G), (-1, 1))
        - 2 * G
    )
    return D


# compute dist, but batched over the graph dim and the alphas dim
batched_compute_dist = torch.vmap(torch.vmap(compute_dist))


def compute_diffusion_from_dist(W, sigma, threshold, mask):
    W = torch.exp(-W / sigma)
    W = torch.where(W < threshold, 0.0, W)
    # Mask has shape (B, N)
    # We first want to broadcast to (B, 1, N)
    # We then want to set any row or column that is masked out to zero
    W = torch.where(mask[:, None, :, None] & mask[:, None, None, :], W, 0.0)
    # We clamp the min to avoid division by zero
    d = W.sum(2, keepdim=True).clamp_min(1e-8)
    W.div_(d)
    # Add self-loops with weight 0.5
    W.diagonal(dim1=-2, dim2=-1).add_(0.5)
    return W


def compute_diffusion_matrix(
    point_clouds: torch.Tensor,
    alphas: torch.Tensor,
    sigma,
    threshold,
    mask: torch.Tensor,
    use_alphas_for_connectivity_only=False,
):
    """Given a batch of point clouds and a set of alphas, compute the diffusion matrices.

    point_clouds: (B, N, d)
    alphas: (n_weights, d)
    mask: (B, N) boolean mask for valid points

    Returns:
    W: (B, n_weights, N, N) diffusion matrices
    X_bar: (B, n_weights, N, d) reweighted point clouds
    """
    # X_bar shape: (B, n_weights, N, d)
    X_bar = point_clouds.unsqueeze(1) * alphas[None, :, None, :]
    W = batched_compute_dist(X_bar)
    W = compute_diffusion_from_dist(W, sigma, threshold, mask)
    if use_alphas_for_connectivity_only:
        # Instead of X_bar, just add the n_weights dimension and use the point clouds
        return W, point_clouds.unsqueeze(1).expand(-1, alphas.shape[0], -1, -1)
    return W, X_bar


class GraphFeatLearningLayer(nn.Module):
    def __init__(
        self,
        n_weights: int,
        dimension: int,
        threshold: float,
        sigma: int,
        J: int,
        device,
        pooling: bool,
        normalize_alphas: bool,
        ignore_alphas: bool,
        use_alphas_for_connectivity_only: bool,
    ):
        super().__init__()

        self.n_weights = n_weights
        self.dimension = dimension
        self.threshold = threshold
        self.device = device
        self.gwt = GraphWaveletTransform(J, device, pooling=pooling)
        self.sigma = sigma
        self.normalize_alphas = normalize_alphas
        self.ignore_alphas = ignore_alphas
        self.use_alphas_for_connectivity_only = use_alphas_for_connectivity_only

        if self.ignore_alphas:
            self.alphas = nn.Parameter(
                torch.ones((n_weights, dimension)).to(device),
                requires_grad=False,
            )
        else:
            self.alphas = nn.Parameter(
                torch.rand((n_weights, dimension)).to(device),
                requires_grad=True,
            )

        assert pooling or (self.n_weights == 1), (
            "n_weights > 1 not supported without pooling"
        )

    def forward(self, point_clouds, mask):
        if self.normalize_alphas:
            # When alpha entries normally distributed, they have norm ~ sqrt(dimension)
            # In order to avoid having alpha -> 0, we normalize the entries to keep the norm fixed at sqrt(dimension)
            norm_value = self.dimension**0.5
            alphas = norm_value * self.alphas / self.alphas.norm(dim=1, keepdim=True)
        else:
            alphas = self.alphas
        W, X_bar = compute_diffusion_matrix(
            point_clouds,
            alphas,
            self.sigma,
            self.threshold,
            mask,
            self.use_alphas_for_connectivity_only,
        )
        # Mask has shape (B, N), expand to (B, n_weights, N) to match W and X_bar
        features = self.gwt(
            W, X_bar, mask.unsqueeze(1).expand((-1, self.n_weights, -1))
        )
        if self.gwt.pooling:
            # Reshape to (B, n_weights * feature_dim)
            return features.view(features.size(0), -1)
        else:
            # If no pooling, we remove the n_weights dimension (which should be 1)
            # We then only select the nodes according to the mask
            # This gives a tensor of shape (sum(num_points_i), feature_dim)
            # where num_points_i is the number of valid points in point_clouds[i] (or equivalently mask[i].sum())
            return features.squeeze(1)[mask]


class SimplicialFeatLearningLayerTri(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device, pooling):
        super().__init__()
        # shape = [n_weights, dimension], each row i is alpha_i \in R^dimension
        self.alphas = nn.Parameter(
            torch.rand((n_weights, dimension), requires_grad=True).to(device)
        )
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device
        self.pooling = pooling

    def forward(self, point_clouds, sigma):
        B_pc = len(point_clouds)
        dim = point_clouds[0].shape[1]

        all_edge_indices = []
        all_edge_weights = []
        all_features = []

        batch = []

        node_offset = 0
        self.indices = []

        for p in range(B_pc):
            pc = point_clouds[p]
            N_pts = pc.shape[0]
            for w in range(self.n_weights):
                alpha_w = self.alphas[w]
                X_nodes = pc * alpha_w

                W = compute_dist(X_nodes)
                W = torch.exp(-W / sigma)

                i_idx, j_idx = torch.where(W >= self.threshold)
                all_edge_indices.append(torch.stack([i_idx, j_idx]))
                edge_weights_ij = W[i_idx, j_idx]
                all_edge_weights.append(edge_weights_ij)
                num_edges = i_idx.shape[0]

                W_thresh = W >= self.threshold
                neighbors = [set() for _ in range(N_pts)]

                i_idx, j_idx = torch.where(W_thresh)
                for i, j in zip(i_idx.tolist(), j_idx.tolist()):
                    if i < j:
                        neighbors[i].add(j)
                        neighbors[j].add(i)

                triangles = []
                for i in range(N_pts):
                    for j in neighbors[i]:
                        if j > i:
                            common_neighbors = neighbors[i].intersection(neighbors[j])
                            for k in common_neighbors:
                                if k > j:
                                    triangles.append((i, j, k))

                valid_tri = torch.tensor(triangles, device=self.device)[
                    :1000
                ]  # shape [?, 3]
                num_tri = valid_tri.size(0)

                X_edges = 0.5 * (X_nodes[i_idx] + X_nodes[j_idx])
                if num_tri:
                    X_tri = (
                        X_nodes[valid_tri[:, 0]]
                        + X_nodes[valid_tri[:, 1]]
                        + X_nodes[valid_tri[:, 2]]
                    ) / 3.0
                    X_bar = torch.cat([X_nodes, X_edges, X_tri], dim=0)
                else:
                    X_bar = torch.cat([X_nodes, X_edges], dim=0)
                index = {}
                edges = torch.stack((i_idx, j_idx)).T
                for k, v in enumerate(edges.tolist()):
                    index[frozenset(v)] = k

                edge_pairs = []
                for e1 in index.keys():
                    for e2 in index.keys():
                        if len(e1.intersection(e2)) == 1:
                            edge_pairs.append([index[e1], index[e2]])
                            edge_pairs.append([index[e2], index[e1]])

                index = {}
                for k, v in enumerate(valid_tri.tolist()):
                    index[frozenset(v)] = k
                tri_pairs = []
                for t1 in index.keys():
                    for t2 in index.keys():
                        if len(t1.intersection(t2)) == 2:
                            tri_pairs.append([index[t1], index[t2]])
                            tri_pairs.append([index[t2], index[t1]])

                base_nodes = node_offset
                base_edges = node_offset + N_pts
                base_tris = node_offset + N_pts + num_edges
                edge_pairs_tensor = torch.tensor(
                    edge_pairs, dtype=torch.long, device=self.device
                )
                edge_pairs_tensor = torch.unique(edge_pairs_tensor, dim=0)
                all_edge_indices.append(edge_pairs_tensor.T + base_edges)
                all_edge_weights.append(
                    edge_weights_ij[edge_pairs_tensor.T[0]]
                    + edge_weights_ij[edge_pairs_tensor.T[1]]
                )

                if num_tri:
                    tri_pairs_tensor = torch.tensor(
                        tri_pairs, dtype=torch.long, device=self.device
                    )
                    all_edge_indices.append(tri_pairs_tensor.T + base_tris)
                    all_edge_weights.append(
                        torch.ones(
                            len(tri_pairs), dtype=torch.float, device=self.device
                        )
                    )
                all_features.append(X_bar)

                n_total = N_pts + num_edges + num_tri
                batch.extend([p * self.n_weights + w] * n_total)

                node_offset += n_total

        edge_index = []
        edge_weight = []
        for i, w in zip(all_edge_indices, all_edge_weights):
            edge_index.append(i)
            edge_weight.append(w)

        edge_index_cat = (
            torch.cat(edge_index, dim=1)
            if len(edge_index) > 0
            else torch.empty((2, 0), device=self.device)
        )
        edge_weight_cat = (
            torch.cat(edge_weight, dim=0)
            if len(edge_weight) > 0
            else torch.empty((0,), device=self.device)
        )

        X_cat = (
            torch.cat(all_features, dim=0)
            if all_features
            else torch.empty((0, dim), device=self.device)
        )
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        J = 3
        gwt = GraphWaveletTransform(
            edge_index_cat, edge_weight_cat, X_cat, J, self.device, self.pooling
        )

        features = gwt.generate_timepoint_features(batch)
        return features.view(B_pc, features.shape[1] * self.n_weights)


class SimplicialFeatLearningLayerTetra(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device, pooling):
        super().__init__()
        # shape = [n_weights, dimension], each row i is alpha_i \in R^dimension
        self.alphas = nn.Parameter(
            torch.rand((n_weights, dimension), requires_grad=True).to(device)
        )
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device
        self.pooling = pooling

    def forward(self, point_clouds, sigma):
        B_pc = len(point_clouds)
        dim = point_clouds[0].shape[1]

        all_edge_indices = []
        all_edge_weights = []
        all_features = []

        batch = []

        node_offset = 0
        self.indices = []

        for p in range(B_pc):
            pc = point_clouds[p]
            N_pts = pc.shape[0]
            for w in range(self.n_weights):
                alpha_w = self.alphas[w]
                X_nodes = pc * alpha_w

                W = compute_dist(X_nodes)
                W = torch.exp(-W / sigma)

                i_idx, j_idx = torch.where(W >= self.threshold)
                all_edge_indices.append(torch.stack([i_idx, j_idx]))
                edge_weights_ij = W[i_idx, j_idx]
                all_edge_weights.append(edge_weights_ij)
                num_edges = i_idx.shape[0]

                W_thresh = W >= self.threshold
                neighbors = [set() for _ in range(N_pts)]

                i_idx, j_idx = torch.where(W_thresh)
                for i, j in zip(i_idx.tolist(), j_idx.tolist()):
                    if i < j:
                        neighbors[i].add(j)
                        neighbors[j].add(i)

                triangles = []
                for i in range(N_pts):
                    for j in neighbors[i]:
                        if j > i:
                            common_neighbors = neighbors[i].intersection(neighbors[j])
                            for k in common_neighbors:
                                if k > j:
                                    triangles.append((i, j, k))

                valid_tri = torch.tensor(triangles, device=self.device)[
                    :1000
                ]  # shape [?, 3]
                num_tri = valid_tri.size(0)

                X_edges = 0.5 * (X_nodes[i_idx] + X_nodes[j_idx])
                if num_tri > 0:
                    X_tri = (
                        X_nodes[valid_tri[:, 0]]
                        + X_nodes[valid_tri[:, 1]]
                        + X_nodes[valid_tri[:, 2]]
                    ) / 3.0
                else:
                    X_tri = torch.empty((0, dim), device=self.device)

                tetrahedra = []
                tri_neighbors = [set() for _ in range(N_pts)]

                for i, j, k in triangles:
                    # i<j<k from how we formed them
                    # intersection:
                    c1 = neighbors[i].intersection(neighbors[j])
                    c2 = neighbors[j].intersection(neighbors[k])
                    c3 = neighbors[i].intersection(neighbors[k])
                    # potential 4th nodes are in intersection of c1, c2, c3
                    # i.e. any node l in c1 ∩ c2 ∩ c3 => i,j,k,l is a tetrahedron
                    common_nbrs_ijk = c1.intersection(c2).intersection(c3)
                    for l in common_nbrs_ijk:
                        if l > k:
                            tetrahedra.append((i, j, k, l))

                valid_tetra = torch.tensor(
                    tetrahedra, device=self.device, dtype=torch.long
                )
                num_tetra = valid_tetra.size(0)

                # 6) Create tetrahedron centroids
                if num_tetra > 0:
                    X_tetra = (
                        X_nodes[valid_tetra[:, 0]]
                        + X_nodes[valid_tetra[:, 1]]
                        + X_nodes[valid_tetra[:, 2]]
                        + X_nodes[valid_tetra[:, 3]]
                    ) / 4.0
                else:
                    X_tetra = torch.empty((0, dim), device=self.device)
                X_bar = torch.cat([X_nodes, X_edges, X_tri, X_tetra], dim=0)

                index = {}
                edges = torch.stack((i_idx, j_idx)).T
                for k, v in enumerate(edges.tolist()):
                    index[frozenset(v)] = k

                edge_pairs = []
                for e1 in index.keys():
                    for e2 in index.keys():
                        if len(e1.intersection(e2)) == 1:
                            edge_pairs.append([index[e1], index[e2]])
                            edge_pairs.append([index[e2], index[e1]])

                index = {}
                for k, v in enumerate(valid_tri.tolist()):
                    index[frozenset(v)] = k
                tri_pairs = []
                for t1 in index.keys():
                    for t2 in index.keys():
                        if len(t1.intersection(t2)) == 2:
                            tri_pairs.append([index[t1], index[t2]])
                            tri_pairs.append([index[t2], index[t1]])

                base_nodes = node_offset
                base_edges = node_offset + N_pts
                base_tris = base_edges + num_edges
                base_tetra = base_tris + num_tri

                original_edges = torch.stack([i_idx, j_idx], dim=0) + base_nodes
                all_edge_indices.append(original_edges)
                all_edge_weights.append(edge_weights_ij)

                if num_tri:
                    tri_pairs_tensor = torch.tensor(
                        tri_pairs, dtype=torch.long, device=self.device
                    )
                    all_edge_indices.append(tri_pairs_tensor.T + base_tris)
                    all_edge_weights.append(
                        torch.ones(
                            len(tri_pairs), dtype=torch.float, device=self.device
                        )
                    )
                all_features.append(X_bar)

                tetra_index = {}
                for idx_t, quadruple in enumerate(valid_tetra.tolist()):
                    tetra_index[frozenset(quadruple)] = idx_t
                tetra_pairs = []
                # Compare each pair of tetrahedra, check if they share 3 vertices
                # In practice, you'd want a more efficient approach than O(num_tetra^2).
                for t1 in tetra_index.keys():
                    for t2 in tetra_index.keys():
                        if t1 != t2 and len(t1.intersection(t2)) == 3:
                            tetra_pairs.append([tetra_index[t1], tetra_index[t2]])

                if len(tetra_pairs) > 0:
                    tetra_pairs_tensor = torch.tensor(
                        tetra_pairs, dtype=torch.long, device=self.device
                    )
                    # shift them by base_tetra
                    tetra_pairs_tensor = tetra_pairs_tensor + base_tetra
                    # adjacency for tetrahedra
                    # all_edge_indices is 2 x E
                    all_edge_indices.append(tetra_pairs_tensor.T)
                    all_edge_weights.append(
                        torch.ones(tetra_pairs_tensor.size(0), device=self.device)
                    )

                n_total = N_pts + num_edges + num_tri + num_tetra
                batch.extend([p * self.n_weights + w] * n_total)

                node_offset += n_total

        edge_index = []
        edge_weight = []
        for i, w in zip(all_edge_indices, all_edge_weights):
            edge_index.append(i)
            edge_weight.append(w)

        edge_index_cat = (
            torch.cat(edge_index, dim=1)
            if len(edge_index) > 0
            else torch.empty((2, 0), device=self.device)
        )
        edge_weight_cat = (
            torch.cat(edge_weight, dim=0)
            if len(edge_weight) > 0
            else torch.empty((0,), device=self.device)
        )

        X_cat = (
            torch.cat(all_features, dim=0)
            if all_features
            else torch.empty((0, dim), device=self.device)
        )
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        J = 3
        gwt = GraphWaveletTransform(
            edge_index_cat, edge_weight_cat, X_cat, J, self.device, self.pooling
        )

        features = gwt.generate_timepoint_features(batch)
        return features.view(B_pc, features.shape[1] * self.n_weights)


class HiPoNet(nn.Module):
    def __init__(
        self,
        dimension,
        n_weights,
        threshold,
        K,
        J,
        device,
        sigma,
        pooling=True,
        normalize_alphas=False,
        ignore_alphas=False,
        use_alphas_for_connectivity_only=False,
    ):
        super(HiPoNet, self).__init__()
        self.dimension = dimension
        if K == 1:
            self.layer = GraphFeatLearningLayer(
                n_weights,
                dimension,
                threshold,
                sigma,
                J,
                device,
                pooling=pooling,
                normalize_alphas=normalize_alphas,
                ignore_alphas=ignore_alphas,
                use_alphas_for_connectivity_only=use_alphas_for_connectivity_only,
            )
        elif K == 2:
            self.layer = SimplicialFeatLearningLayerTri(
                n_weights, dimension, threshold, sigma, J, device, pooling=pooling
            )
        else:
            self.layer = SimplicialFeatLearningLayerTetra(
                n_weights, dimension, threshold, sigma, J, device, pooling=pooling
            )
        self.device = device
        self.sigma = sigma

    def forward(self, batch, mask):
        return self.layer(batch, mask)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bn=True):
        super(MLP, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim) if bn else None
        if num_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, X):
        if self.bn:
            X = self.bn(X)
        for i in range(len(self.layers) - 1):
            X = F.relu(self.layers[i](X))
        return self.layers[-1](X)


class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, bn=False):
        super(MLPAutoEncoder, self).__init__()
        self.encoder = MLP(input_dim, hidden_dim, embedding_dim, num_layers, bn)
        self.decoder = MLP(embedding_dim, hidden_dim, input_dim, num_layers, bn)

    def encode(self, X):
        return F.relu(self.encoder(X))

    def decode(self, X):
        return self.decoder(X)

    def forward(self, X):
        return self.decoder(self.encoder(X))
