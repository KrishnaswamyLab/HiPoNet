import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GWT import GraphWaveletTransform
from models.SWT import SimplicialWaveletTransform, PoolingAttention
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

@torch.vmap
def get_D_matrix(G):
    return (
        torch.reshape(torch.diag(G), (1, -1))
        + torch.reshape(torch.diag(G), (-1, 1))
        - 2 * G
    )


# Need to implement manually to use nested tensors
def single_batched_compute_dist(X):
    G = torch.matmul(X, X.permute(0, 2, 1))
    D = get_D_matrix(G)
    return D


# compute dist, but batched over the graph dim and the alphas dim
# double_batched_compute_dist = torch.vmap(torch.vmap(single_batched_compute_dist))
double_batched_compute_dist = torch.vmap(torch.vmap(compute_dist))


def normalized_population_distance_loss(
    z: torch.Tensor,
    target_dists: torch.Tensor,
) -> torch.Tensor:
    """Match relative population distances without depending on batch scale."""
    if z.shape[0] < 3:
        return z.new_zeros(())
    latent_dists = torch.pdist(z)
    if latent_dists.numel() != target_dists.numel():
        raise ValueError(
            f"Expected {latent_dists.numel()} target distances, got {target_dists.numel()}"
        )
    latent_scale = latent_dists.detach().mean().clamp_min(1e-8)
    target_scale = target_dists.detach().mean().clamp_min(1e-8)
    return F.smooth_l1_loss(latent_dists / latent_scale, target_dists / target_scale)


def masked_mse_reconstruction_loss(
    recon_points: torch.Tensor,
    target_points: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Padding-aware index-wise reconstruction loss."""
    point_mask = mask.unsqueeze(-1).float()
    sq_err = (recon_points - target_points).pow(2)
    denom = (point_mask.sum() * target_points.shape[-1]).clamp_min(1.0)
    return (sq_err * point_mask).sum() / denom


def masked_chamfer_loss(
    recon_points: torch.Tensor,
    target_points: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric squared Chamfer loss with padding masked out on targets.

    The decoder emits a fixed-size point cloud, so every reconstructed point is
    treated as active. Target padding is excluded from both nearest-neighbor
    directions.
    """
    target_mask = mask.bool()
    dists = torch.cdist(recon_points, target_points, p=2).pow(2)
    large = torch.finfo(dists.dtype).max

    recon_to_target = dists.masked_fill(~target_mask[:, None, :], large).min(dim=2).values
    target_to_recon = dists.min(dim=1).values

    valid_counts = target_mask.sum(dim=1).clamp_min(1).to(dists.dtype)
    target_term = (target_to_recon * target_mask.to(dists.dtype)).sum(dim=1) / valid_counts
    recon_term = recon_to_target.mean(dim=1)
    return (recon_term + target_term).mean()


def point_cloud_reconstruction_loss(
    recon_points: torch.Tensor,
    point_clouds: torch.Tensor,
    mask: torch.Tensor,
    max_points: int,
    point_dim: int,
    loss_type: str,
) -> torch.Tensor:
    """Compute padded point-cloud reconstruction loss."""
    target_points = point_clouds[:, :max_points, :point_dim]
    target_mask = mask[:, :max_points]
    if loss_type == "mse":
        return masked_mse_reconstruction_loss(recon_points, target_points, target_mask)
    if loss_type == "chamfer":
        return masked_chamfer_loss(recon_points, target_points, target_mask)
    raise ValueError(f"Unknown reconstruction loss: {loss_type}")


def pairwise_masked_chamfer_distances(
    point_clouds: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Upper-triangular pairwise squared Chamfer distances for a batch.

    The returned vector uses the same pair order as ``torch.pdist``. It is used
    to regularize population-level geometry after decoding, so generated
    populations keep roughly the same spread as the real populations in the
    batch.
    """
    batch_size = point_clouds.shape[0]
    if batch_size < 2:
        return point_clouds.new_zeros((0,))

    mask = mask.bool()
    large = torch.finfo(point_clouds.dtype).max
    distances = []
    for i in range(batch_size - 1):
        xi = point_clouds[i : i + 1]
        mi = mask[i : i + 1]
        ni = mi.sum().clamp_min(1).to(point_clouds.dtype)
        for j in range(i + 1, batch_size):
            xj = point_clouds[j : j + 1]
            mj = mask[j : j + 1]
            nj = mj.sum().clamp_min(1).to(point_clouds.dtype)
            dists = torch.cdist(xi, xj, p=2).pow(2)
            i_to_j = dists.masked_fill(~mj[:, None, :], large).min(dim=2).values
            j_to_i = dists.masked_fill(~mi[:, :, None], large).min(dim=1).values
            chamfer = (i_to_j * mi.to(dists.dtype)).sum() / ni
            chamfer = chamfer + (j_to_i * mj.to(dists.dtype)).sum() / nj
            distances.append(chamfer)
    return torch.stack(distances)


def decoded_population_distance_loss(
    recon_points: torch.Tensor,
    point_clouds: torch.Tensor,
    mask: torch.Tensor,
    max_points: int,
    point_dim: int,
) -> torch.Tensor:
    """Match pairwise population distances before and after decoding."""
    if recon_points.shape[0] < 2:
        return recon_points.new_zeros(())

    target_points = point_clouds[:, :max_points, :point_dim]
    target_mask = mask[:, :max_points]
    recon_mask = torch.ones(
        recon_points.shape[:2], dtype=torch.bool, device=recon_points.device
    )

    recon_d = pairwise_masked_chamfer_distances(recon_points, recon_mask)
    with torch.no_grad():
        target_d = pairwise_masked_chamfer_distances(target_points, target_mask)
        target_d = target_d / target_d.max().clamp(min=1e-8)
    recon_d = recon_d / recon_d.detach().max().clamp(min=1e-8)
    return F.mse_loss(recon_d, target_d)


def compute_diffusion_from_dist(W, sigma, threshold, mask):
    W = torch.exp(-W / sigma)
    W = torch.where(W < threshold, 0.0, W)
    if len(W.shape) == 4:
        # Mask has shape (B, N)
        # We first want to broadcast to (B, 1, N)
        W_mask = mask[:, None, :, None] & mask[:, None, None, :]
    else:
        W_mask = mask[:, :, None] & mask[:, None, :]
    # We then want to set any row or column that is masked out to zero
    W = torch.where(W_mask, W, 0.0)
    # We clamp the min to avoid division by zero
    d = W.sum(2, keepdim=True).clamp_min(1e-8)
    W.div_(d)
    # Add self-loops with weight 0.5
    W.diagonal(dim1=-2, dim2=-1).add_(0.5)
    return W


def compute_diffusion_matrix(
    point_clouds: torch.Tensor,
    alphas: torch.Tensor | None,
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
    X_bar = point_clouds
    if alphas is not None:
        X_bar = X_bar.unsqueeze(1)
        X_bar = X_bar * alphas[None, :, None, :]
        W = double_batched_compute_dist(X_bar)
    else:
        # Don't need to batch over alphas dim
        W = single_batched_compute_dist(X_bar)
    W = compute_diffusion_from_dist(W, sigma, threshold, mask)
    if use_alphas_for_connectivity_only and alphas is not None:
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
        softmax_alphas: bool,
    ):
        super().__init__()

        if ignore_alphas:
            n_weights = 1

        self.n_weights = n_weights
        self.dimension = dimension
        self.threshold = threshold
        self.device = device
        self.gwt = GraphWaveletTransform(J, device, pooling=pooling)
        self.sigma = sigma
        self.normalize_alphas = normalize_alphas
        self.ignore_alphas = ignore_alphas
        self.use_alphas_for_connectivity_only = use_alphas_for_connectivity_only
        self.softmax_alphas = softmax_alphas

        if self.normalize_alphas and self.softmax_alphas:
            raise ValueError(
                "Only one of normalize_alphas and softmax_alphas can be True"
            )

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

    def forward(self, point_clouds, mask, node_features=None):
        if self.ignore_alphas:
            alphas = None
        elif self.normalize_alphas:
            # When alpha entries normally distributed, they have norm ~ sqrt(dimension)
            # In order to avoid having alpha -> 0, we normalize the entries to keep the norm fixed at sqrt(dimension)
            norm_value = self.dimension**0.5
            alphas = norm_value * self.alphas / self.alphas.norm(dim=1, keepdim=True)
        elif self.softmax_alphas:
            alphas = torch.nn.functional.softmax(self.alphas, dim=1)
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
        # If separate node features are provided (e.g. gene expression for a
        # spatial view where coords only define the graph), use them instead.
        if node_features is not None:
            X_bar = node_features.unsqueeze(1).expand(-1, self.n_weights, -1, -1)

        if alphas is not None:
            # Mask has shape (B, N), expand to (B, n_weights, N) to match W and X_bar
            mask = mask.unsqueeze(1).expand((-1, self.n_weights, -1))

        features = self.gwt(W, X_bar, mask)
        if self.gwt.pooling:
            # Reshape to (B, n_weights * feature_dim)
            return features.view(features.size(0), -1)
        else:
            # If no pooling, we remove the n_weights dimension (which should be 1)
            # We then only select the nodes according to the mask
            # This gives a tensor of shape (sum(num_points_i), feature_dim)
            # where num_points_i is the number of valid points in point_clouds[i] (or equivalently mask[i].sum())
            return features.squeeze(1)[mask]

    def forward_with_W(self, point_clouds, mask, node_features=None):
        """Like forward() but also returns the diffusion matrix W for the GW loss.

        Returns
        -------
        features : as returned by forward()
        W        : (B, n_weights, N, N) diffusion operators (or (B, N, N) when
                   ignore_alphas is True)
        mask_out : the (possibly expanded) mask used internally
        """
        if self.ignore_alphas:
            alphas = None
        elif self.normalize_alphas:
            norm_value = self.dimension**0.5
            alphas = norm_value * self.alphas / self.alphas.norm(dim=1, keepdim=True)
        elif self.softmax_alphas:
            alphas = torch.nn.functional.softmax(self.alphas, dim=1)
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
        if node_features is not None:
            X_bar = node_features.unsqueeze(1).expand(-1, self.n_weights, -1, -1)
        mask_out = mask
        if alphas is not None:
            mask_out = mask.unsqueeze(1).expand((-1, self.n_weights, -1))
        features = self.gwt(W, X_bar, mask_out)
        if self.gwt.pooling:
            return features.view(features.size(0), -1), W, mask
        else:
            return features.squeeze(1)[mask_out], W, mask


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


class SimplicialFeatLearningLayer(nn.Module):
    """Unified simplicial feature learning layer using SWT with proper Hodge Laplacians.

    Replaces the broken SimplicialFeatLearningLayerTri/Tetra with a correct
    implementation that uses the SimplicialWaveletTransform pipeline.

    Supports both combinatorial and geometric (metric-aware) Hodge Laplacians.

    Parameters
    ----------
    n_weights : int
        Number of learnable alpha weight vectors.
    dimension : int
        Feature dimension of input point clouds.
    threshold : float
        Threshold for graph/complex construction.
    sigma : float
        Gaussian kernel bandwidth.
    J : int
        Number of wavelet scales.
    device : torch.device or str
    pooling : bool
        If True, pool features across nodes.
    max_simplex_dim : int
        Maximum simplex dimension (2=triangles, 3=tetrahedra).
    use_geometric_laplacian : bool
        If True, use diffusion distances + Cayley-Menger mass matrices for
        the geometric Hodge Laplacian. Fully differentiable w.r.t. alphas.
    diffusion_steps : int
        Number of steps t for P^t when computing diffusion distances.
    """

    def __init__(
        self,
        n_weights,
        dimension,
        threshold,
        sigma,
        J,
        device,
        pooling=True,
        max_simplex_dim=2,
        use_geometric_laplacian=False,
        diffusion_steps=1,
        use_attention=False,
    ):
        super().__init__()
        self.alphas = nn.Parameter(
            torch.rand((n_weights, dimension), requires_grad=True).to(device)
        )
        self.n_weights = n_weights
        self.dimension = dimension
        self.threshold = threshold
        self.sigma = sigma
        self.J = J
        self.device = device
        self.pooling = pooling
        self.max_simplex_dim = max_simplex_dim
        self.use_geometric_laplacian = use_geometric_laplacian
        self.diffusion_steps = diffusion_steps
        self.use_attention = use_attention

        if use_attention:
            self.pool_attn = nn.ModuleList(
                [PoolingAttention(dimension).to(device) for _ in range(max_simplex_dim + 1)]
            )
        else:
            self.pool_attn = None

    @staticmethod
    def _normalize_operator(op: torch.Tensor | None) -> torch.Tensor | None:
        """Convert a square operator to a row-stochastic matrix for GW costs."""
        if op is None or op.numel() == 0:
            return None
        if op.dim() != 2 or op.shape[0] != op.shape[1] or op.shape[0] < 2:
            return None
        sym_op = 0.5 * (op + op.t())
        sym_op = sym_op.clamp_min(0.0)
        d = sym_op.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return sym_op / d

    def _extract_simplex_operators(self, adj: torch.Tensor, swt: SimplicialWaveletTransform) -> list[torch.Tensor]:
        """Build a compact set of simplex-level operators used by GW regularisation."""
        ops: list[torch.Tensor] = []

        node_op = self._normalize_operator(adj)
        if node_op is not None:
            ops.append(node_op)

        edge_candidates = []
        if len(swt.P_L) > 1 and swt.P_L[1] is not None:
            edge_candidates.append(swt.P_L[1])
        if len(swt.P_U) > 1 and swt.P_U[1] is not None:
            edge_candidates.append(swt.P_U[1])
        if edge_candidates:
            edge_op = self._normalize_operator(sum(edge_candidates) / len(edge_candidates))
            if edge_op is not None:
                ops.append(edge_op)

        tri_candidates = []
        if len(swt.P_L) > 2 and swt.P_L[2] is not None:
            tri_candidates.append(swt.P_L[2])
        if len(swt.P_U) > 2 and swt.P_U[2] is not None:
            tri_candidates.append(swt.P_U[2])
        if tri_candidates:
            tri_op = self._normalize_operator(sum(tri_candidates) / len(tri_candidates))
            if tri_op is not None:
                ops.append(tri_op)

        return ops

    def forward(self, point_clouds, mask, node_features=None, return_structure: bool = False):
        """Compute simplicial wavelet features for a batch of point clouds.

        Parameters
        ----------
        point_clouds : torch.Tensor, shape (B, N, d_coord)
            Padded batch of point clouds used to build the graph/complex topology.
        mask : torch.Tensor, shape (B, N)
            Boolean mask for valid (non-padded) points.
        node_features : torch.Tensor, shape (B, N, d_feat), optional
            If provided, these are used as node signals in the wavelet transform
            instead of the (alpha-reweighted) coordinates. Useful for spatial
            views where coords define topology but gene expression is the signal.

        Returns
        -------
        features : torch.Tensor, shape (B, n_weights * feature_dim)
            Pooled simplicial wavelet features.
        """
        B_pc = point_clouds.shape[0]
        all_features = []
        all_structures = [] if return_structure else None

        for p in range(B_pc):
            # Extract valid points for this sample
            valid = mask[p]
            pc = point_clouds[p][valid]  # (N_pts, d_coord)
            nf = node_features[p][valid] if node_features is not None else None
            weight_features = []
            sample_structures = [] if return_structure else None

            for w in range(self.n_weights):
                alpha_w = self.alphas[w]
                X_w = pc * alpha_w  # used for graph topology

                # Node signal: gene features (if provided) else coord-based X_w
                X_signal = nf if nf is not None else X_w

                # Build adjacency matrix via Gaussian kernel + threshold
                D = compute_dist(X_w)
                W = torch.exp(-D / self.sigma)
                adj = torch.where(W >= self.threshold, W, torch.zeros_like(W))

                # Compute squared diffusion distances if using geometric Laplacian.
                # Row-normalize adj -> P, raise to diffusion_steps, then take
                # pairwise L2 distances between rows. Fully differentiable.
                sq_diff_dists = None
                if self.use_geometric_laplacian:
                    d = adj.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    P = adj / d
                    Pt = torch.linalg.matrix_power(P, self.diffusion_steps)
                    sq_diff_dists = compute_dist(Pt)

                # Build SWT and compute wavelet coefficients
                swt = SimplicialWaveletTransform(
                    adj,
                    X_signal,
                    self.threshold,
                    self.device,
                    use_geometric_laplacian=self.use_geometric_laplacian,
                    sq_diff_dists=sq_diff_dists,
                )
                coeff = swt.calculate_wavelet_coeff(self.J, pool_attn=self.pool_attn)
                weight_features.append(coeff)
                if return_structure:
                    sample_structures.append(self._extract_simplex_operators(adj, swt))

            # Concatenate features across weights
            all_features.append(torch.cat(weight_features))
            if return_structure:
                all_structures.append(sample_structures)

        # Stack batch
        features = torch.stack(all_features, dim=0)  # (B, n_weights * feat_dim)
        if return_structure:
            return features, all_structures
        return features

    def forward_with_structure(self, point_clouds, mask, node_features=None):
        """Forward pass returning GW-ready simplicial operators."""
        return self.forward(
            point_clouds,
            mask,
            node_features=node_features,
            return_structure=True,
        )

    def get_attention_weights(self):
        """Return last attention weights per simplex dimension for interpretability.

        Returns
        -------
        dict[int, torch.Tensor | None]
            Maps simplex dimension to attention weights from the most recent
            forward pass, or None if attention is disabled.
        """
        if self.pool_attn is None:
            return {}
        return {k: attn.last_weights for k, attn in enumerate(self.pool_attn)}


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
        softmax_alphas=False,
        use_geometric_laplacian=False,
        diffusion_steps=1,
        use_attention=False,
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
                softmax_alphas=softmax_alphas,
            )
        elif K >= 2:
            self.layer = SimplicialFeatLearningLayer(
                n_weights,
                dimension,
                threshold,
                sigma,
                J,
                device,
                pooling=pooling,
                max_simplex_dim=K,
                use_geometric_laplacian=use_geometric_laplacian,
                diffusion_steps=diffusion_steps,
                use_attention=use_attention,
            )
        self.device = device
        self.sigma = sigma

    def forward(self, batch, mask, node_features=None):
        return self.layer(batch, mask, node_features=node_features)

    def forward_with_W(self, batch, mask, node_features=None):
        """Forward pass that also returns the diffusion matrix W.

        Only supported when K=1 (``GraphFeatLearningLayer``).  Raises
        ``NotImplementedError`` for simplicial layers.
        """
        if not isinstance(self.layer, GraphFeatLearningLayer):
            raise NotImplementedError(
                "forward_with_W is only supported for K=1 (GraphFeatLearningLayer)."
            )
        return self.layer.forward_with_W(batch, mask, node_features=node_features)

    def forward_with_structure(self, batch, mask, node_features=None):
        """Forward pass that also returns GW-ready structure for any K."""
        if isinstance(self.layer, GraphFeatLearningLayer):
            feats, W, base_mask = self.layer.forward_with_W(
                batch, mask, node_features=node_features
            )
            return feats, {"type": "graph", "W": W, "mask": base_mask}

        feats, simplex_ops = self.layer.forward_with_structure(
            batch, mask, node_features=node_features
        )
        return feats, {"type": "simplicial", "operators": simplex_ops}

    def get_attention_weights(self):
        """Proxy to layer's attention weights (K>=2 only)."""
        if hasattr(self.layer, 'get_attention_weights'):
            return self.layer.get_attention_weights()
        return {}


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
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_layers,
        bn=False,
        output_dim=None,
    ):
        super(MLPAutoEncoder, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.encoder = MLP(input_dim, hidden_dim, embedding_dim, num_layers, bn)
        self.decoder = MLP(embedding_dim, hidden_dim, output_dim, num_layers, bn)

    def encode(self, X):
        return F.tanh(self.encoder(X))

    def decode(self, X):
        return self.decoder(X)

    def forward(self, X):
        return self.decoder(self.encoder(X))


# ---------------------------------------------------------------------------
# Unsupervised HiPoNet autoencoder
# ---------------------------------------------------------------------------

class HiPoNetAutoencoder(nn.Module):
    """Unsupervised autoencoder built on top of HiPoNet.

    Pipeline
    --------
    1. **HiPoNet** computes fixed-size graph-level wavelet features
       ``(B, wavelet_dim)`` (pooling must be enabled).
     2. **Encoder MLP**: ``wavelet_dim → latent_dim``.
     3. **Decoder MLP**: ``latent_dim → (max_points * point_dim)`` reshaped to
         ``(B, max_points, point_dim)``.

    Loss
    ----
    ``total = recon_loss + dist_weight * dist_loss + decoded_dist_weight * decoded_dist_loss``

    * ``recon_loss`` — Masked reconstruction loss between decoded and original
      padded point clouds.
    * ``dist_loss`` — Stress loss: normalised pairwise Euclidean distances in
      latent space should match normalised precomputed UDEMD distances between
      point clouds. Targets must be passed to ``compute_loss`` as
      ``target_dists`` (upper-triangular vector, same order as
      ``torch.pdist``).
    * ``decoded_dist_loss`` — Stress loss between pairwise decoded-population
      Chamfer distances and pairwise real-population Chamfer distances.

    Parameters
    ----------
    hiponet : HiPoNet
        A *pooling-enabled* HiPoNet instance (call ``HiPoNet(..., pooling=True)``).
    wavelet_dim : int
        Dimensionality of the HiPoNet output (inferred at construction time via
        a dummy forward pass, or passed explicitly).
    latent_dim : int
        Dimensionality of the latent embedding space.
    hidden_dims : list[int] | None
        Hidden layer widths for encoder/decoder MLPs. Default: ``[256, 128]``.
    point_dim : int
        Dimensionality of each point.
    max_points : int
        Maximum number of points in the padded representation.
    dist_weight : float
        λ for the UDEMD distance preservation term. Set to 0 to disable.
    decoded_dist_weight : float
        λ for preserving pairwise decoded-population distances. Set to 0 to disable.
    """

    def __init__(
        self,
        hiponet: "HiPoNet",
        wavelet_dim: int,
        latent_dim: int,
        hidden_dims: list | None = None,
        point_dim: int = 0,
        max_points: int = 0,
        dist_weight: float = 0.1,
        decoded_dist_weight: float = 0.0,
        recon_loss_type: str = "mse",
    ):
        super().__init__()
        self.hiponet = hiponet
        self.dist_weight = dist_weight
        self.decoded_dist_weight = decoded_dist_weight
        self.point_dim = point_dim
        self.max_points = max_points
        self.recon_loss_type = recon_loss_type

        if self.point_dim <= 0 or self.max_points <= 0:
            raise ValueError("point_dim and max_points must both be positive")

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Encoder: wavelet_dim → latent_dim
        enc: list[nn.Module] = []
        in_d = wavelet_dim
        for h in hidden_dims:
            enc += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        enc.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc)

        # Decoder: latent_dim → max_points * point_dim (mirrored)
        dec: list[nn.Module] = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        dec.append(nn.Linear(in_d, max_points * point_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, point_clouds, mask, node_features=None):
        """Return ``(z, feats)`` — latent codes and raw wavelet features."""
        feats = self.hiponet(point_clouds, mask, node_features=node_features)
        z = self.encoder(feats)
        return z, feats

    def forward(self, point_clouds, mask, node_features=None):
        """Return ``(z, recon_points, feats)``."""
        z, feats = self.encode(point_clouds, mask, node_features=node_features)
        recon = self.decoder(z).view(-1, self.max_points, self.point_dim)
        return z, recon, feats

    def _dist_loss(self, z: torch.Tensor, target_dists: torch.Tensor) -> torch.Tensor:
        """Stress loss: MSE between normalised pairwise latent distances and
        precomputed UDEMD distances (upper-triangular vector, same order as
        ``torch.pdist``).
        """
        return normalized_population_distance_loss(z, target_dists)

    def compute_loss(
        self,
        point_clouds: torch.Tensor,
        mask: torch.Tensor,
        target_dists: torch.Tensor | None = None,
        node_features=None,
    ):
        """Full unsupervised loss.

        Parameters
        ----------
        point_clouds : (B, N, d)
        mask         : (B, N)
        target_dists : 1-D tensor of length B*(B-1)/2 — precomputed UDEMD
                       distances for all pairs in the batch, in the same
                       upper-triangular order as ``torch.pdist``.  Required
                       when ``dist_weight > 0``.
        node_features: optional node signals

        Returns
        -------
        (total_loss, recon_loss, dist_loss, decoded_dist_loss, kl_loss) — scalar tensors
        """
        z, recon_points, _ = self(point_clouds, mask, node_features=node_features)

        recon_loss = point_cloud_reconstruction_loss(
            recon_points,
            point_clouds,
            mask,
            self.max_points,
            self.point_dim,
            self.recon_loss_type,
        )

        if self.dist_weight > 0.0 and target_dists is not None:
            dist_loss = self._dist_loss(z, target_dists)
        else:
            dist_loss = recon_points.new_zeros(())

        if self.decoded_dist_weight > 0.0:
            decoded_dist_loss = decoded_population_distance_loss(
                recon_points,
                point_clouds,
                mask,
                self.max_points,
                self.point_dim,
            )
        else:
            decoded_dist_loss = recon_points.new_zeros(())

        kl_loss = recon_points.new_zeros(())
        total = (
            recon_loss
            + self.dist_weight * dist_loss
            + self.decoded_dist_weight * decoded_dist_loss
        )
        return total, recon_loss, dist_loss, decoded_dist_loss, kl_loss


class HiPoNetWaveletAutoencoder(nn.Module):
    """Autoencoder that reconstructs HiPoNet wavelet features, not point clouds.

    The latent representation is trained from the population-level wavelet
    function produced by HiPoNet. A separate generation head can learn to map
    latent codes back to cells, but its reconstruction loss is applied to
    ``z.detach()`` so point generation does not shape the encoder.
    """

    def __init__(
        self,
        hiponet: "HiPoNet",
        wavelet_dim: int,
        latent_dim: int,
        hidden_dims: list | None = None,
        point_dim: int = 0,
        max_points: int = 0,
        dist_weight: float = 0.1,
        generator_weight: float = 0.0,
        recon_loss_type: str = "chamfer",
    ):
        super().__init__()
        self.hiponet = hiponet
        self.wavelet_dim = wavelet_dim
        self.latent_dim = latent_dim
        self.dist_weight = dist_weight
        self.generator_weight = generator_weight
        self.point_dim = point_dim
        self.max_points = max_points
        self.recon_loss_type = recon_loss_type

        if self.generator_weight > 0.0 and (self.point_dim <= 0 or self.max_points <= 0):
            raise ValueError(
                "point_dim and max_points must be positive when generator_weight > 0"
            )

        if hidden_dims is None:
            hidden_dims = [256, 128]

        enc: list[nn.Module] = []
        in_d = wavelet_dim
        for h in hidden_dims:
            enc += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        enc.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc)

        wave_dec: list[nn.Module] = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            wave_dec += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        wave_dec.append(nn.Linear(in_d, wavelet_dim))
        self.wavelet_decoder = nn.Sequential(*wave_dec)

        self.generation_head = None
        if self.generator_weight > 0.0:
            gen: list[nn.Module] = []
            in_d = latent_dim
            for h in reversed(hidden_dims):
                gen += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
                in_d = h
            gen.append(nn.Linear(in_d, max_points * point_dim))
            self.generation_head = nn.Sequential(*gen)

    def encode(self, point_clouds, mask, node_features=None):
        feats = self.hiponet(point_clouds, mask, node_features=node_features)
        z = self.encoder(feats)
        return z, feats

    def reconstruct_wavelet(self, z: torch.Tensor) -> torch.Tensor:
        return self.wavelet_decoder(z)

    def generate_points(self, z: torch.Tensor) -> torch.Tensor:
        if self.generation_head is None:
            raise RuntimeError("Point generation head is disabled; set generator_weight > 0")
        return self.generation_head(z).view(-1, self.max_points, self.point_dim)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.generate_points(z)

    def forward(self, point_clouds, mask, node_features=None):
        z, feats = self.encode(point_clouds, mask, node_features=node_features)
        recon_feats = self.reconstruct_wavelet(z)
        gen_points = (
            self.generate_points(z.detach()) if self.generation_head is not None else None
        )
        return z, recon_feats, gen_points, feats

    def _dist_loss(self, z: torch.Tensor, target_dists: torch.Tensor) -> torch.Tensor:
        return normalized_population_distance_loss(z, target_dists)

    def compute_loss(
        self,
        point_clouds: torch.Tensor,
        mask: torch.Tensor,
        target_dists: torch.Tensor | None = None,
        node_features=None,
    ):
        z, feats = self.encode(point_clouds, mask, node_features=node_features)
        recon_feats = self.reconstruct_wavelet(z)
        recon_loss = F.mse_loss(recon_feats, feats.detach())

        if self.dist_weight > 0.0 and target_dists is not None:
            dist_loss = self._dist_loss(z, target_dists)
        else:
            dist_loss = recon_feats.new_zeros(())

        if self.generator_weight > 0.0:
            gen_points = self.generate_points(z.detach())
            generator_loss = point_cloud_reconstruction_loss(
                gen_points,
                point_clouds,
                mask,
                self.max_points,
                self.point_dim,
                self.recon_loss_type,
            )
        else:
            generator_loss = recon_feats.new_zeros(())

        kl_loss = recon_feats.new_zeros(())
        total = recon_loss + self.dist_weight * dist_loss + self.generator_weight * generator_loss
        return total, recon_loss, dist_loss, generator_loss, kl_loss


class HiPoNetVAE(nn.Module):
    """Variational autoencoder built on top of HiPoNet graph-level features.

    Compared with ``HiPoNetAutoencoder``, the encoder predicts a Gaussian
    posterior ``q(z | point_cloud) = N(mu, diag(exp(logvar)))``. Training uses
    the reparameterization trick and adds a KL penalty toward ``N(0, I)`` so
    random latent samples can be decoded into point clouds.
    """

    def __init__(
        self,
        hiponet: "HiPoNet",
        wavelet_dim: int,
        latent_dim: int,
        hidden_dims: list | None = None,
        point_dim: int = 0,
        max_points: int = 0,
        dist_weight: float = 0.1,
        kl_weight: float = 1.0,
        recon_loss_type: str = "chamfer",
    ):
        super().__init__()
        self.hiponet = hiponet
        self.dist_weight = dist_weight
        self.kl_weight = kl_weight
        self.point_dim = point_dim
        self.max_points = max_points
        self.latent_dim = latent_dim
        self.recon_loss_type = recon_loss_type

        if self.point_dim <= 0 or self.max_points <= 0:
            raise ValueError("point_dim and max_points must both be positive")

        if hidden_dims is None:
            hidden_dims = [256, 128]

        enc: list[nn.Module] = []
        in_d = wavelet_dim
        for h in hidden_dims:
            enc += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        self.encoder = nn.Sequential(*enc)
        self.mu_head = nn.Linear(in_d, latent_dim)
        self.logvar_head = nn.Linear(in_d, latent_dim)

        dec: list[nn.Module] = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        dec.append(nn.Linear(in_d, max_points * point_dim))
        self.decoder = nn.Sequential(*dec)

    def encode_distribution(self, point_clouds, mask, node_features=None):
        """Return ``(mu, logvar, feats)`` for q(z | point_cloud)."""
        feats = self.hiponet(point_clouds, mask, node_features=node_features)
        h = self.encoder(feats)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=-12.0, max=12.0)
        return mu, logvar, feats

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + eps * sigma using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, point_clouds, mask, node_features=None, sample: bool = False):
        """Return latent codes and raw HiPoNet features.

        By default this returns ``mu`` for stable embeddings. Set
        ``sample=True`` to draw from the approximate posterior.
        """
        mu, logvar, feats = self.encode_distribution(
            point_clouds, mask, node_features=node_features
        )
        z = self.reparameterize(mu, logvar) if sample else mu
        return z, feats

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into padded point clouds."""
        return self.decoder(z).view(-1, self.max_points, self.point_dim)

    def sample(self, n_samples: int, device=None) -> torch.Tensor:
        """Sample z ~ N(0, I) and decode to generated point clouds."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode_latent(z)

    def forward(self, point_clouds, mask, node_features=None):
        """Return ``(z, recon_points, feats)`` using posterior sampling."""
        mu, logvar, feats = self.encode_distribution(
            point_clouds, mask, node_features=node_features
        )
        z = self.reparameterize(mu, logvar)
        recon = self.decode_latent(z)
        return z, recon, feats

    def _dist_loss(self, z: torch.Tensor, target_dists: torch.Tensor) -> torch.Tensor:
        if z.shape[0] < 2:
            return z.new_zeros(())
        lat_d = torch.pdist(z)
        lat_d_n = lat_d / lat_d.detach().max().clamp(min=1e-8)
        tgt_n = target_dists / target_dists.max().clamp(min=1e-8)
        return F.mse_loss(lat_d_n, tgt_n)

    @staticmethod
    def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Mean KL per sample for q(z|x) || N(0, I).
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return kl.mean()

    def compute_loss(
        self,
        point_clouds: torch.Tensor,
        mask: torch.Tensor,
        target_dists: torch.Tensor | None = None,
        node_features=None,
        kl_weight: float | None = None,
    ):
        """Return ``(total_loss, recon_loss, dist_loss, kl_loss)``."""
        mu, logvar, _ = self.encode_distribution(
            point_clouds, mask, node_features=node_features
        )
        z = self.reparameterize(mu, logvar)
        recon_points = self.decode_latent(z)

        recon_loss = point_cloud_reconstruction_loss(
            recon_points,
            point_clouds,
            mask,
            self.max_points,
            self.point_dim,
            self.recon_loss_type,
        )

        if self.dist_weight > 0.0 and target_dists is not None:
            dist_loss = self._dist_loss(mu, target_dists)
        else:
            dist_loss = recon_points.new_zeros(())

        kl_loss = self._kl_loss(mu, logvar)
        beta = self.kl_weight if kl_weight is None else kl_weight
        total = recon_loss + self.dist_weight * dist_loss + beta * kl_loss
        return total, recon_loss, dist_loss, kl_loss
