import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import defaultdict

from models.GWT import GraphWaveletTransform
from models.SWT import SimplicialWaveletTransform
import gc
gc.enable()


def compute_dist(X):
    G = torch.matmul(X, X.T)
    D = torch.reshape(torch.diag(G), (1, -1)) + torch.reshape(torch.diag(G), (-1, 1)) - 2 * G
    return D
class GraphFeatLearningLayer(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device):
        super().__init__()
        self.alphas = nn.Parameter(torch.rand((n_weights, dimension), requires_grad=True).to(device))
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device

    def forward(self, point_clouds, sigma):
        B_pc = len(point_clouds)
        d = point_clouds[0].shape[1]
        
        all_edge_indices = []
        all_edge_weights = []
        all_node_feats   = []
        
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
                W = W/d
                # W[torch.isnan(W)] = 0
                # W = 1/2*(torch.eye(W.shape[0]).to(self.device)+W)
                
                row, col = torch.where(W > 0)
                w_vals   = W[row, col]

                row_offset = row + node_offset
                col_offset = col + node_offset
                all_edge_indices.append(torch.cat([torch.stack([row_offset, col_offset], dim=0), torch.arange(W.shape[0]).repeat(2,1).to(W.device)], 1))
                all_edge_weights.append(torch.cat([w_vals/2, 0.5*torch.ones(W.shape[0]).to(W.device)]))

                all_node_feats.append(X_bar)

                batch.extend([p*self.n_weights + i]*num_points)

                node_offset += num_points

        edge_index = torch.cat(all_edge_indices, dim=1).to(self.device)  
        edge_weight = torch.cat(all_edge_weights, dim=0).to(self.device) 
        X_cat = torch.cat(all_node_feats, dim=0).to(self.device)

        batch = torch.tensor(batch, device=self.device, dtype=torch.long)

        J = 3
        gwt = GraphWaveletTransform(edge_index, edge_weight, X_cat, J, self.device)

        features = gwt.diffusion_only(batch)
        return features.view(B_pc, features.shape[1] * self.n_weights)
class SimplicialFeatLearningLayerTri(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device):
        super().__init__()
        # shape = [n_weights, dimension], each row i is alpha_i \in R^dimension
        self.alphas = nn.Parameter(torch.rand((n_weights, dimension), requires_grad=True).to(device))
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device

    def forward(self, point_clouds, sigma):
        B_pc = len(point_clouds)
        dim = point_clouds[0].shape[1]

        all_edge_indices = []
        all_edge_weights = []
        all_features     = []

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

                W_thresh = (W >= self.threshold)
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

                valid_tri = torch.tensor(triangles, device=self.device)[:1000]  # shape [?, 3]
                num_tri = valid_tri.size(0)

                X_edges = 0.5 * ( X_nodes[i_idx] + X_nodes[j_idx] )
                if(num_tri):
                    X_tri = (X_nodes[valid_tri[:,0]] +
                            X_nodes[valid_tri[:,1]] +
                            X_nodes[valid_tri[:,2]]) / 3.0
                    X_bar = torch.cat([X_nodes, X_edges, X_tri], dim=0)  
                else:
                    X_bar = torch.cat([X_nodes, X_edges], dim=0)  
                index = {}
                edges = torch.stack((i_idx,j_idx)).T
                for k,v in enumerate(edges.tolist()):
                    index[frozenset(v)] = k

                edge_pairs = []
                for e1 in index.keys():
                    for e2 in index.keys():
                        if(len(e1.intersection(e2)) == 1):
                            edge_pairs.append([index[e1], index[e2]])
                            edge_pairs.append([index[e2], index[e1]])
                
                index = {}
                for k,v in enumerate(valid_tri.tolist()):
                    index[frozenset(v)] = k
                tri_pairs = []
                for t1 in index.keys():
                    for t2 in index.keys():
                        if(len(t1.intersection(t2)) == 2):
                            tri_pairs.append([index[t1], index[t2]])
                            tri_pairs.append([index[t2], index[t1]])

                base_nodes = node_offset
                base_edges = node_offset + N_pts
                base_tris  = node_offset + N_pts + num_edges
                edge_pairs_tensor = torch.tensor(edge_pairs, dtype=torch.long, device=self.device)
                edge_pairs_tensor = torch.unique(edge_pairs_tensor, dim=0)
                all_edge_indices.append(edge_pairs_tensor.T + base_edges)
                all_edge_weights.append(edge_weights_ij[edge_pairs_tensor.T[0]] + edge_weights_ij[edge_pairs_tensor.T[1]])

                if(num_tri):
                    tri_pairs_tensor = torch.tensor(tri_pairs, dtype=torch.long, device=self.device)
                    all_edge_indices.append(tri_pairs_tensor.T + base_tris)
                    all_edge_weights.append(torch.ones(len(tri_pairs), dtype=torch.float, device=self.device))
                all_features.append(X_bar)

                n_total = N_pts + num_edges + num_tri
                batch.extend([p*self.n_weights + w]*n_total)

                node_offset += n_total

        edge_index = []
        edge_weight = []
        for i, w in zip(all_edge_indices, all_edge_weights):
            edge_index.append(i)
            edge_weight.append(w)

        edge_index_cat = torch.cat(edge_index, dim=1) if len(edge_index)>0 else torch.empty((2,0), device=self.device)
        edge_weight_cat = torch.cat(edge_weight, dim=0) if len(edge_weight)>0 else torch.empty((0,), device=self.device)

        X_cat = torch.cat(all_features, dim=0) if all_features else torch.empty((0, dim), device=self.device)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        J = 3
        gwt = GraphWaveletTransform(edge_index_cat, edge_weight_cat, X_cat, J, self.device)

        features = gwt.generate_timepoint_features(batch)
        return features.view(B_pc, features.shape[1] * self.n_weights)

class SimplicialFeatLearningLayerTetra(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device):
        super().__init__()
        # shape = [n_weights, dimension], each row i is alpha_i \in R^dimension
        self.alphas = nn.Parameter(torch.rand((n_weights, dimension), requires_grad=True).to(device))
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device

    def forward(self, point_clouds, sigma):
        B_pc = len(point_clouds)
        dim = point_clouds[0].shape[1]

        all_edge_indices = []
        all_edge_weights = []
        all_features     = []

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

                W_thresh = (W >= self.threshold)
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

                valid_tri = torch.tensor(triangles, device=self.device)[:1000]  # shape [?, 3]
                num_tri = valid_tri.size(0)

                X_edges = 0.5 * ( X_nodes[i_idx] + X_nodes[j_idx] )
                if num_tri > 0:
                    X_tri = (
                        X_nodes[valid_tri[:, 0]] +
                        X_nodes[valid_tri[:, 1]] +
                        X_nodes[valid_tri[:, 2]]
                    ) / 3.0
                else:
                    X_tri = torch.empty((0, dim), device=self.device)
                    
                tetrahedra = []
                tri_neighbors = [set() for _ in range(N_pts)]
                
                for (i, j, k) in triangles:
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

                valid_tetra = torch.tensor(tetrahedra, device=self.device, dtype=torch.long)
                num_tetra   = valid_tetra.size(0)

                # 6) Create tetrahedron centroids
                if num_tetra > 0:
                    X_tetra = (
                        X_nodes[valid_tetra[:, 0]] +
                        X_nodes[valid_tetra[:, 1]] +
                        X_nodes[valid_tetra[:, 2]] +
                        X_nodes[valid_tetra[:, 3]]
                    ) / 4.0
                else:
                    X_tetra = torch.empty((0, dim), device=self.device)
                X_bar = torch.cat([X_nodes, X_edges, X_tri, X_tetra], dim=0)

                index = {}
                edges = torch.stack((i_idx,j_idx)).T
                for k,v in enumerate(edges.tolist()):
                    index[frozenset(v)] = k

                
                edge_pairs = []
                for e1 in index.keys():
                    for e2 in index.keys():
                        if(len(e1.intersection(e2)) == 1):
                            edge_pairs.append([index[e1], index[e2]])
                            edge_pairs.append([index[e2], index[e1]])
                
                index = {}
                for k,v in enumerate(valid_tri.tolist()):
                    index[frozenset(v)] = k
                tri_pairs = []
                for t1 in index.keys():
                    for t2 in index.keys():
                        if(len(t1.intersection(t2)) == 2):
                            tri_pairs.append([index[t1], index[t2]])
                            tri_pairs.append([index[t2], index[t1]])

                base_nodes = node_offset
                base_edges = node_offset + N_pts
                base_tris  = base_edges + num_edges
                base_tetra = base_tris  + num_tri
                
                original_edges = torch.stack([i_idx, j_idx], dim=0) + base_nodes
                all_edge_indices.append(original_edges)
                all_edge_weights.append(edge_weights_ij)


                if(num_tri):
                    tri_pairs_tensor = torch.tensor(tri_pairs, dtype=torch.long, device=self.device)
                    all_edge_indices.append(tri_pairs_tensor.T + base_tris)
                    all_edge_weights.append(torch.ones(len(tri_pairs), dtype=torch.float, device=self.device))
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
                    tetra_pairs_tensor = torch.tensor(tetra_pairs, dtype=torch.long, device=self.device)
                    # shift them by base_tetra
                    tetra_pairs_tensor = tetra_pairs_tensor + base_tetra
                    # adjacency for tetrahedra
                    # all_edge_indices is 2 x E
                    all_edge_indices.append(tetra_pairs_tensor.T)
                    all_edge_weights.append(torch.ones(tetra_pairs_tensor.size(0), device=self.device))



                n_total = N_pts + num_edges + num_tri + num_tetra
                batch.extend([p*self.n_weights + w]*n_total)

                node_offset += n_total


        edge_index = []
        edge_weight = []
        for i, w in zip(all_edge_indices, all_edge_weights):
            edge_index.append(i)
            edge_weight.append(w)

        edge_index_cat = torch.cat(edge_index, dim=1) if len(edge_index)>0 else torch.empty((2,0), device=self.device)
        edge_weight_cat = torch.cat(edge_weight, dim=0) if len(edge_weight)>0 else torch.empty((0,), device=self.device)

        X_cat = torch.cat(all_features, dim=0) if all_features else torch.empty((0, dim), device=self.device)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        J = 3
        gwt = GraphWaveletTransform(edge_index_cat, edge_weight_cat, X_cat, J, self.device)

        features = gwt.generate_timepoint_features(batch)
        return features.view(B_pc, features.shape[1] * self.n_weights)

class HiPoNet(nn.Module):
    def __init__(self, dimension, n_weights, threshold, K, device, sigma):
        super(HiPoNet, self).__init__()
        self.dimension = dimension
        if K == 1:
            self.layer = GraphFeatLearningLayer(n_weights, dimension, threshold, device)
        elif K ==2:          
            self.layer = SimplicialFeatLearningLayerTri(n_weights, dimension, threshold, device)
        else:
            self.layer = SimplicialFeatLearningLayerTetra(n_weights, dimension, threshold, device)
        self.device = device
        self.sigma = sigma
    
    def forward(self, batch):
        PSI = self.layer(batch, self.sigma)
        return PSI

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.sf = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(input_dim)
        if(num_layers==1):
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for i in range(num_layers-2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, X):
        X = self.bn(X)
        for i in range(len(self.layers)-1):
            X = F.relu(self.layers[i](X))
        return (self.layers[-1](X))
    
class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers):
        self.encoder = MLP(input_dim, hidden_dim, embedding_dim, num_layers)
        self.decoder = MLP(embedding_dim, hidden_dim, input_dim, num_layers)

    def encode(self, X):
        return F.relu(self.encoder(X))
    
    def decode(self, X):
        return self.decoder(X)
