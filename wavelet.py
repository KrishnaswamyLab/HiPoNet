import argparse
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool

from utils.read_data import load_data

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.n_weights = 2
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

        self.max_scale = 2 ** (J - 1)
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


class DenseGraphWaveletTransform:
    """
    This class is used to generate graph wavelet transform features from a given adjacency matrix and node features.
    The graph wavelet transform is a method to generate features from a graph that are invariant to the graph's structure."""

    def __init__(self, adj, ro, device):
        self.adj = adj
        self.ro = ro
        self.device = device
        d = self.adj.sum(0)
        P_t = self.adj / d
        P_t[torch.isnan(P_t)] = 0
        self.P = 1 / 2 * (torch.eye(P_t.shape[0]).to(self.device) + P_t)
        self.psi = []
        for d1 in [1, 2, 4, 8, 16]:
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
        u1 = torch.zeros((self.ro.shape)).to(self.device)
        for j in range(len(self.psi)):
            for j_prime in range(0, j):
                if j_prime == 0 and j == 0:
                    u1 = torch.abs(self.psi[j_prime] @ u[j])
                else:
                    u1 = torch.cat((u1, torch.abs(self.psi[j_prime] @ u[j])), 1)
        return u1

    def generate_timepoint_feature(self):
        F0 = self.zero_order_feature()
        F1, u = self.first_order_feature()
        F2 = self.second_order_feature(u)
        F = torch.concatenate((F0, F1), axis=1)
        F = torch.concatenate((F, F2), axis=1)
        return F


def sparse_forward(point_clouds):
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

    J = 3
    gwt = GraphWaveletTransform(edge_index, edge_weight, X_cat, J, self.device)

    features = gwt.generate_timepoint_features(batch)
    features = features.view(B_pc, features.shape[1] * self.n_weights)
    return features


def dense_forward(point_clouds):
    self = args
    PSI = []
    for point_cloud in point_clouds:
        for i in range(self.n_weights):
            X_bar = (point_cloud)*self.alphas[i]
            W = compute_dist(X_bar)
            W = torch.exp(-(W / 10))
            W = torch.where(W < self.threshold, torch.zeros_like(W), W)
            gwt = DenseGraphWaveletTransform(W, X_bar, self.device)
            PSI.append(gwt.generate_timepoint_feature())
    return torch.cat(PSI, dim = 1)


def main(args):
    alphas = torch.rand((args.n_weights, 44), requires_grad=True).to(args.device)
    args.alphas = alphas

    PCs, labels, num_labels = load_data(
        "/home/tl855/project_pi_sk2433/shared/Hiren_2025_HiPoNet/pdo_data/", ""
    )

    input_tensor = torch.nested.as_nested_tensor(
        PCs[:8], device=args.device, layout=torch.jagged
    )

    with torch.profiler.profile(
        [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p1:
        sparse_out = sparse_forward(input_tensor)
        loss = sparse_out.sum()
        loss.backward()

    # Reset grads
    alphas.grad.zero_()

    with torch.profiler.profile(
        [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p2:
        dense_out = dense_forward(input_tensor)
        loss = sparse_out.sum()
        loss.backward()

    p1.export_chrome_trace("sparse.json")
    p2.export_chrome_trace("dense.json")

    if not torch.allclose(sparse_out, dense_out):
        raise ValueError("sparse and dense should give same results")


if __name__ == "__main__":
    main(args)
