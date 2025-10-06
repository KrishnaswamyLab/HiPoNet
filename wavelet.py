import argparse
import torch
import time

from models.graph_learning import compute_dist, GraphFeatLearningLayer
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
    alphas = torch.rand((args.n_weights, 44)).to(args.device)
    alphas.requires_grad_(True)
    args.alphas = alphas

    PCs, labels, num_labels = load_data(
        "/home/tl855/project_pi_sk2433/shared/Hiren_2025_HiPoNet/pdo_data/", ""
    )

    PCs = sorted(PCs, key=lambda x: x.shape[0], reverse=True)

    input_tensor = torch.nested.as_nested_tensor(
        [p[: args.num_points] for p in PCs[:4]], device=args.device, layout=torch.jagged
    ).to_padded_tensor(padding=0.0)
    mask = input_tensor.sum(-1) != 0
    print("input shape:", input_tensor.shape)

    timings = []
    max_memory = []

    gwt = GraphFeatLearningLayer(
        args.n_weights, 44, args.threshold, args.sigma, args.J, args.device, pooling=True
    ).to(args.device)
    gwt.compile(fullgraph=True)
    # Run it once to JIT compile
    sparse_out_new = gwt(input_tensor, mask)

    N_repeats = 10

    t, m = [], []
    for _ in range(N_repeats):
        start = time.time()
        sparse_out_new = gwt(input_tensor, mask)
        loss = sparse_out_new.sum()
        loss.backward()
        end = time.time()
        t.append(end - start)
        m.append(torch.cuda.max_memory_allocated() / GB)
        # Reset grads
        gwt.alphas.grad.zero_()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    timings.append(
        f"sparse_new {torch.tensor(t).mean().item():.4f}s +- {torch.std(torch.tensor(t)).item():.4f}s"
    )
    max_memory.append(
        f"sparse_new {torch.tensor(m).mean().item():.4f}GB +- {torch.std(torch.tensor(m)).item():.4f}GB"
    )

    t, m = [], []
    for _ in range(N_repeats):
        start = time.time()
        dense_out = dense_forward(input_tensor, args.J)
        loss = dense_out.sum()
        loss.backward()
        end = time.time()
        t.append(end - start)
        m.append(torch.cuda.max_memory_allocated() / GB)

        # Reset grads
        alphas.grad.zero_()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    timings.append(
        f"dense {torch.tensor(t).mean().item():.4f}s +- {torch.std(torch.tensor(t)).item():.4f}s"
    )
    max_memory.append(
        f"dense {torch.tensor(m).mean().item():.4f}GB +- {torch.std(torch.tensor(m)).item():.4f}GB"
    )

    print("-----------")
    print(sparse_out_new.shape, dense_out.shape)
    print("-----------")
    print(max_memory)
    print(timings)


if __name__ == "__main__":
    main(args)
