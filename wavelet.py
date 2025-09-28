import argparse
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter

from utils.read_data import load_data

GB = 1024**3

parser = argparse.ArgumentParser()
parser.add_argument("--n_weights", type=int, default=2)
parser.add_argument("-n", "--num_points", type=int, default=-1)
parser.add_argument("-J", type=int, default=3)
args = parser.parse_args()
args.threshold = 0.15
args.sigma = 10


if torch.cuda.is_available():
    args.device = "cuda"
else:
    args.device = "cpu"


def compute_dist(X):
    G = torch.matmul(X, X.T)
    D = (
        torch.reshape(torch.diag(G), (1, -1))
        + torch.reshape(torch.diag(G), (-1, 1))
        - 2 * G
    )
    return D


batched_compute_dist = torch.vmap(torch.vmap(compute_dist))


class WeightedSumConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(-1)


class GraphWaveletTransform(nn.Module):
    def __init__(self, edge_index, edge_weight, X, J, device, pooling: bool = True):
        super().__init__()
        self.device = device
        # We'll store the graph
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device)
        self.X_init = X.to(device)  # node features

        self.conv = WeightedSumConv()
        self.J = J
        self.num_feats = self.X_init.size(1)

        self.max_scale = 2**J
        self.pooling = pooling

    def diffuse(self, x=None):
        if x is None:
            x = self.X_init

        x_curr = x
        out_list = []
        for step in range(1, self.max_scale + 1):
            x_curr = self.conv(x_curr, self.edge_index, self.edge_weight)
            if (step & (step - 1)) == 0:
                out_list.append(x_curr)
        return out_list

    def first_order_feature(self, diff_list):
        F1 = torch.cat(
            [
                torch.abs(diff_list[i - 1] - diff_list[i])
                for i in range(1, len(diff_list))
            ],
            1,
        )
        return F1

    def second_order_feature(self, diff_list):
        U = torch.cat(diff_list, dim=1)
        U_diff_list = self.diffuse(U)

        results = []
        for j in range(self.J):
            col_start = j * self.num_feats
            col_end = (j + 1) * self.num_feats
            for j_prime in range(j + 1, self.J):
                block_jp = U_diff_list[j_prime][:, col_start:col_end]
                block_jp_1 = U_diff_list[j_prime - 1][:, col_start:col_end]
                results.append(torch.abs(block_jp - block_jp_1))
        return torch.cat(results, dim=1)

    def generate_timepoint_features(self, batch):
        diff_list = self.diffuse()
        F0 = diff_list[-1]
        F1 = self.first_order_feature(diff_list)
        F2 = self.second_order_feature(diff_list)
        feats = torch.cat([F0, F1, F2], dim=1)

        if self.pooling:
            feats = global_mean_pool(feats, batch)

        return feats

    def diffusion_only(self, batch):
        diff_list = self.diffuse()  # list of length J

        feats = torch.cat(diff_list, dim=1)

        if self.pooling:
            feats = global_mean_pool(feats, batch)

        return feats


class SparseGraphWaveletTransform(nn.Module):
    def __init__(self, J, device, pooling: bool = True):
        super().__init__()
        self.device = device
        self.J = J
        self.pooling = pooling

    def generate_timepoint_features(self, P, X):
        num_points = P.shape[0]
        P_powered = P
        F1 = []
        F2 = []
        PSI = torch.zeros(size=(0, num_points, num_points), device=self.device)
        for i in range(self.J):
            # We multiply P_powered with itself, giving powers of 2: P^2 = PP, P^4 = (P^2)(P^2), ...
            new_P_powered: torch.Tensor = torch.mm(P_powered, P_powered)
            # The wavelet operator is the *difference* between diffusion scales i+1 and i
            psi_i = new_P_powered - P_powered
            # The first-order scattering coefficient is given by psi X, followed by nonlinearity
            # F1 accumulates first-order features
            scattering_coef_i = torch.abs(torch.mm(psi_i, X)).unsqueeze(0)
            F1.append(scattering_coef_i)
            F2.append(
                torch.abs(
                    torch.matmul(PSI, scattering_coef_i.expand(PSI.shape[0], -1, -1))
                )
            )

            # We accumulate the different scales into a single operator for scales 1 to i+1
            # Note that we make this a *tensor*, since we want to use it to calculate
            # second-order features |psi_i @ |psi_j X| |
            # We only want the *off-diagonal* second-order elements, so we do this after F2
            PSI = torch.cat((PSI, psi_i.unsqueeze(0)), 0)

            # Reset for next loop
            P_powered = new_P_powered

        F0 = torch.mm(P_powered, X).unsqueeze(0)
        features = torch.cat([F0, *F1, *F2])

        if self.pooling:
            features = features.mean(dim=1)

        return features.flatten()

    # Batch over the graphs, and batch over the alphas
    batched_generate_timepoint_features = torch.vmap(
        torch.vmap(generate_timepoint_features, in_dims=(None, 0, 0)),
        in_dims=(None, 0, 0),
    )


class DenseGraphWaveletTransform:
    """
    This class is used to generate graph wavelet transform features from a given adjacency matrix and node features.
    The graph wavelet transform is a method to generate features from a graph that are invariant to the graph's structure."""

    def __init__(self, adj, ro, device, J):
        self.adj = adj
        self.ro = ro
        self.device = device
        d = self.adj.sum(0)
        P_t = self.adj / d
        P_t[torch.isnan(P_t)] = 0
        self.P = 1 / 2 * (torch.eye(P_t.shape[0]).to(self.device) + P_t)
        self.psi = []
        for d1 in [2**i for i in range(0, J)]:
            W_d1 = torch.matrix_power(self.P, d1) - torch.matrix_power(self.P, 2 * d1)
            self.psi.append(W_d1)

    def zero_order_feature(self):
        F0 = torch.matrix_power(self.P, 16) @ self.ro
        return F0

    def first_order_feature(self):
        u = [torch.abs(self.psi[i] @ self.ro) for i in range(len(self.psi))]
        F1 = torch.cat(u, 1)
        return F1, u

    def second_order_feature(self, u):
        features = []
        for j in range(len(self.psi)):
            for j_prime in range(0, j):
                features.append(torch.abs(self.psi[j_prime] @ u[j]))
        return torch.cat(features, dim=1)

    def generate_timepoint_feature(self):
        F0 = self.zero_order_feature()
        F1, u = self.first_order_feature()
        F2 = self.second_order_feature(u)
        F = torch.concatenate((F0, F1), axis=1)
        F = torch.concatenate((F, F2), axis=1)
        return F


def sparse_forward(point_clouds, J):
    self = args
    sigma = 10

    B_pc = len(point_clouds)
    d = point_clouds[0].shape[1]

    all_edge_indices = []
    all_edge_weights = []
    all_node_feats = []

    batch = []

    node_offset = 0

    for p in range(B_pc):
        pc = point_clouds[p]
        num_points = pc.shape[0]
        for i in range(self.n_weights):
            X_bar = pc * self.alphas[i]

            W = compute_dist(X_bar)
            W = torch.exp(-W / sigma)
            W = torch.where(W < self.threshold, torch.zeros_like(W), W)
            d = W.sum(0)
            W = W / d
            # W[torch.isnan(W)] = 0
            # W = 1/2*(torch.eye(W.shape[0]).to(self.device)+W)

            row, col = torch.where(W > 0)
            w_vals = W[row, col]

            row_offset = row + node_offset
            col_offset = col + node_offset
            all_edge_indices.append(
                torch.cat(
                    [
                        torch.stack([row_offset, col_offset], dim=0),
                        (node_offset + torch.arange(W.shape[0]).repeat(2, 1)).to(
                            W.device
                        ),
                    ],
                    1,
                )
            )
            all_edge_weights.append(
                torch.cat([w_vals / 2, 0.5 * torch.ones(W.shape[0]).to(W.device)])
            )

            all_node_feats.append(X_bar)

            batch.extend([p * self.n_weights + i] * num_points)

            node_offset += num_points

    edge_index = torch.cat(all_edge_indices, dim=1).to(self.device)
    edge_weight = torch.cat(all_edge_weights, dim=0).to(self.device)
    X_cat = torch.cat(all_node_feats, dim=0).to(self.device)

    batch = torch.tensor(batch, device=self.device, dtype=torch.long)

    gwt = GraphWaveletTransform(edge_index, edge_weight, X_cat, J, self.device)

    features = gwt.generate_timepoint_features(batch)
    features = features.view(B_pc, features.shape[1] * self.n_weights)
    return features


def sparse_forward_new(point_clouds, J):
    self = args
    sigma = 10

    B_pc = len(point_clouds)
    d = point_clouds[0].shape[1]
    features = []

    gwt = SparseGraphWaveletTransform(J, self.device)

    # X_bar shape: (B, n_weights, N, d)
    X_bar = point_clouds.unsqueeze(1) * self.alphas[None, :, None, :]
    W = batched_compute_dist(X_bar)
    W = torch.exp(-W / sigma)
    W = torch.where(W < self.threshold, torch.zeros_like(W), W)
    d = W.sum(2, keepdim=True)
    W.div_(d)
    W.diagonal(dim1=-2, dim2=-1).add_(0.5)
    gwt.batched_generate_timepoint_features(
        W,
        X_bar,
    )
    return torch.stack(features, 0)


def dense_forward(point_clouds, J):
    self = args
    PSI = []
    for point_cloud in point_clouds:
        for i in range(self.n_weights):
            X_bar = (point_cloud) * self.alphas[i]
            W = compute_dist(X_bar)
            W = torch.exp(-(W / 10))
            W = torch.where(W < self.threshold, torch.zeros_like(W), W)
            gwt = DenseGraphWaveletTransform(W, X_bar, self.device, J)
            feats = gwt.generate_timepoint_feature()
            PSI.append(feats.mean(0))
    return torch.stack(PSI, dim=1)


def main(args):
    import time

    alphas = torch.rand((args.n_weights, 44)).to(args.device)
    alphas.requires_grad_(True)
    args.alphas = alphas

    PCs, labels, num_labels = load_data(
        "/home/tl855/project_pi_sk2433/shared/Hiren_2025_HiPoNet/pdo_data/", ""
    )

    # PCs = sorted(PCs, key=lambda x: x.shape[0], reverse=True)

    input_tensor = torch.nested.as_nested_tensor(
        [p[: args.num_points] for p in PCs[:6]], device=args.device, layout=torch.jagged
    ).to_padded_tensor(padding=0.0)

    timings = []
    max_memory = []

    start = time.time()
    sparse_out_new = sparse_forward_new(input_tensor, args.J)
    loss = sparse_out_new.sum()
    loss.backward()
    end = time.time()
    timings.append(("sparse new", end - start))
    max_memory.append(("post sparse new", torch.cuda.max_memory_allocated() / GB))

    # Reset grads
    alphas.grad.zero_()
    print("Completed sparse new")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    dense_out = dense_forward(input_tensor, args.J)
    loss = dense_out.sum()
    loss.backward()
    end = time.time()
    timings.append(("dense", end - start))
    max_memory.append(("post dense", torch.cuda.max_memory_allocated() / GB))

    print("Completed dense")

    print(sparse_out_new.shape, dense_out.shape)
    print("-----------")
    print(max_memory)
    print(timings)


if __name__ == "__main__":
    main(args)
