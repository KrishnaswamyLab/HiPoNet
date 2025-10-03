import scanpy as sc
import pathlib
import torch
from sklearn.preprocessing import StandardScaler

SEA_LOC = pathlib.Path("/home/tl855/project_pi_sk2433/shared/Hiren_2025_HiPoNet/SEA_ST/sea/")

for file in SEA_LOC.glob("*.h5ad"):
    adata = sc.read_h5ad(file)
    print(f"Loaded file {file.name}.\nNow pre-processing the data.")

    for directory in ["sea_matrix", "sea_spatial", "sea_labels"]:
        pathlib.Path(f"./data/{directory}").mkdir(exist_ok=True)

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X = torch.tensor(StandardScaler().fit_transform(adata.X), dtype=torch.float)
    torch.save(X, f"data/sea_matrix/{file.name}.pt",)
    X_spatial = torch.tensor(StandardScaler().fit_transform(adata.obsm["spatial"]), dtype=torch.float)
    torch.save(X_spatial, f"data/sea_spatial/{file.name}.pt")
    cell_type_labels = torch.tensor(adata.obs.values.squeeze())
    torch.save(cell_type_labels, f"data/sea_labels/{file.name}.pt")

