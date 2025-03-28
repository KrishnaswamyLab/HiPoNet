import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import torch.nn.functional as F
import os.path as osp
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import to_dense_batch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from tqdm import tqdm
import pandas as pd


def load_data(raw_dir, full):
    if raw_dir == "melanoma_data_full":
        if full:
            suffix = '_full'
        else:
            suffix = ''
        with open(os.path.join(raw_dir, 'pc'+suffix+'.pkl'), 'rb') as handle:
            PCs = pickle.load(handle)
            PCs = [torch.tensor(StandardScaler().fit_transform(PCs[i]), dtype=torch.float) for i in range(len(PCs))]
        labels = np.load(os.path.join(raw_dir, 'labels'+suffix+'.npy'))
        num_labels = len(np.unique(labels))
    elif raw_dir == "COVID_data":
        with open(os.path.join(raw_dir, 'filtered_point_clouds_final.pickle'), 'rb') as handle:
            PCs = pickle.load(handle)
            PCs = [torch.tensor(StandardScaler().fit_transform(PCs[i].values[:100]), dtype=torch.float) for i in PCs]
        with open(os.path.join(raw_dir, 'filtered_point_cloud_labels_final.pickle'), 'rb') as handle:
            labels = list(pickle.load(handle).values())
        
        # with open(os.path.join(raw_dir, 'pc_covid.pkl'), 'rb') as handle:
        #     PCs = pickle.load(handle)
        #     PCs = [torch.tensor(StandardScaler().fit_transform(PCs[i]), dtype=torch.float) for i in range(len(PCs))]
        # with open(os.path.join(raw_dir, 'patient_list_covid.pkl'), 'rb') as handle:
        #     subsampled_patient_ids = pickle.load(handle)
        # labels = np.load(os.path.join(raw_dir, 'labels.npy'))
        num_labels = len(np.unique(labels))
    elif raw_dir == "pdo_data":
        with open(osp.join(raw_dir, 'pc_pdo_treatment.pkl'), 'rb') as handle:
            PCs = pickle.load(handle)
            PCs = [torch.tensor(StandardScaler().fit_transform(PCs[i]), dtype=torch.float) for i in range(len(PCs))]
        keep = []
        for i in range(len(PCs)):
            if(PCs[i].shape[0]>200):
                keep.append(i)
        le = LabelEncoder()
        labels = le.fit_transform(np.load(osp.join(raw_dir, 'labels_pdo_treatment.npy')))
        PCs = [PCs[i] for i in keep]
        labels = [labels[i] for i in keep]
        num_labels = len(np.unique(labels))
    return PCs, labels, num_labels

def load_data_persistence(raw_dir, full):
        data = np.load(os.path.join(raw_dir, 'pc_persistence.npy'), allow_pickle = True)

        PCs = [torch.tensor(i['pc'], dtype=torch.float) for i in data]
        h0 = torch.from_numpy(np.vstack([i['h0_bc'] for i in data]))
        h1 = torch.from_numpy(np.vstack([i['h1_bc'][:99] for i in data]))
        labels = F.normalize(torch.cat([h0, h1], 1))
        return PCs, labels, labels.shape[1]

def load_data_ST(raw_dir, label_name):
        spatial_cords = torch.load(f"ST_preprocessed/spatial_cords_{raw_dir}_{label_name}.pt")
        num_pcs = len(spatial_cords)
        gene_expr = torch.load(f"ST_preprocessed/gene_expr_{raw_dir}_{label_name}.pt")
        labels = torch.load(f"ST_preprocessed/labels_{raw_dir}_{label_name}.pt")
        indices = torch.load(f"ST_preprocessed/indices_{raw_dir}_{label_name}.pt")
        spatial_cords = [spatial_cords[i][indices[i]].float() for i in range(num_pcs)]
        gene_expr = [gene_expr[i][indices[i]].float() for i in range(num_pcs)]
        gene_dim = gene_expr[0].shape[1]

        return spatial_cords, gene_expr, labels, 2

def load_data_ST_melanoma(root):
        graph_names = [f[:-3] for f in os.listdir(root) if f.endswith(".pt")]
        graph_files = [f for f in os.listdir(root) if f.endswith(".pt")]
        patient_metadata = pd.read_csv("/gpfs/gibbs/pi/krishnaswamy_smita/hm638/SCGFM/data/Melanoma/patient_info.csv")
        max_num_features = 29
        spatial_cords = []
        gene_expr = []
        labels = []

        print("Preprocessing data!")
        for idx in tqdm(range(len(graph_names))):
            label = patient_metadata[patient_metadata.id == graph_names[idx]].response_binary.values[0]
            graphs = torch.load(os.path.join(root, graph_files[idx]))
            X = []
            for k in range(1, len(graphs)):
                X.append(graphs[k].X.squeeze(1).tolist())
            genes = torch.Tensor(X).float()
            if genes is not None and genes.shape[1] < max_num_features:
                padding = torch.zeros((len(genes), max_num_features - genes.shape[1]))
                genes = torch.cat([genes, padding], dim=1)
            gene_expr.append(genes)
            spatial_cords.append(graphs[0].X.float())
            labels.append(label)
        return spatial_cords, gene_expr, torch.LongTensor(labels), 2