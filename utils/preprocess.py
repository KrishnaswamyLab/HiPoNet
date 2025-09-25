import scanpy as sc
import numpy as np
import pathlib
from glob import glob
import torch
from sklearn.preprocessing import StandardScaler

file = glob("data/sea_raw/*")[0]
file_name = file.split("/")[-1][:-5]
file_path = "data/sea_raw/" + file_name + ".h5ad"
adata = sc.read_h5ad(file_path)
print(f"Loaded file {file_name}.\nNow pre-processing the data.")

for directory in ["sea_braak_labels", "sea_matrix", "sea_spatial"]:
    pathlib.Path(f"./data/{directory}").mkdir(exist_ok=True)


braak_stages = adata.obs["Braak"].values
np.save("data/sea_braak_labels/" + file_name + ".npy", braak_stages)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

X = torch.tensor(StandardScaler().fit_transform(adata.X), dtype=torch.float)
torch.save(X, f"data/sea_matrix/{file_name}.pt",)
X_spatial = torch.tensor(StandardScaler().fit_transform(adata.obsm["X_spatial_raw"]), dtype=torch.float)
torch.save(X_spatial, f"data/sea_spatial/{file_name}.pt")