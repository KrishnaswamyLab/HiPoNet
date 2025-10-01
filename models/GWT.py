import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing


class WeightedSumConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(-1)


class GraphWaveletTransform(nn.Module):
    def __init__(self, J, device, pooling: bool = True):
        super().__init__()
        self.device = device
        self.J = J
        self.pooling = pooling

    def generate_timepoint_features(self, P, X, mask):
        """Generates graph wavelet features.

        There are three types of features:
        - Zeroth-order: P^J X
        - First-order: |psi_j X| for j in 1,...,J where psi_j = P^{2^j} - P^{2^{j-1}}
        - Second-order: |psi_j |psi_i X| | for i < j
        
        P: Transition matrix (num_points x num_points)
        X: Node features (num_points x num_features)
        mask: Mask for valid nodes (num_points,)
        
        """
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
            # We *sum* instead of mean, since we want to ignore masked-out nodes
            features = (features.sum(dim=1) / mask.sum()).flatten()
        else:
            features = features.permute(1, 0, 2).reshape(num_points, -1)
    
        return features

    # Batch over the graphs, and batch over the alphas
    forward = torch.vmap(
        torch.vmap(generate_timepoint_features, in_dims=(None, 0, 0, 0)),
        in_dims=(None, 0, 0, 0),
    )
