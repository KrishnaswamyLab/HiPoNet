import scanpy as sc
import numpy as np
import pathlib
from glob import glob

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

np.save(f"data/sea_matrix/{file_name}.npy", adata.X)
np.save(f"data/sea_spatial/{file_name}.npy", adata.obsm["X_spatial_raw"])