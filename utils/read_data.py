import torch
import pickle
import os
import torch.nn.functional as F
import os.path as osp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import pandas as pd
import pathlib


def _load_kang_pbmc_populations(raw_dir, **kwargs):
    import urllib.request

    import scanpy as sc
    import scipy.sparse as sparse
    from sklearn.decomposition import TruncatedSVD

    root = kwargs.get("kang_root") or raw_dir
    if root in {"kang_pbmc", "kang_ifnb", "pbmc_ifnb"}:
        root = os.path.join("data", "kang_pbmc")
    os.makedirs(root, exist_ok=True)

    h5ad_path = os.path.join(root, "pbmc.h5ad")
    if not os.path.isfile(h5ad_path) or os.path.getsize(h5ad_path) < 100_000_000:
        url = "https://api.figshare.com/v2/file/download/18037739"
        print(f"Downloading Kang PBMC IFN-beta data from {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as response, open(h5ad_path, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    cells_per_population = int(kwargs.get("kang_cells_per_population", 128))
    if cells_per_population < 0:
        raise ValueError("kang_cells_per_population must be non-negative")
    n_populations = kwargs.get("kang_n_populations", kwargs.get("max_samples", 256))
    n_populations = int(n_populations) if n_populations is not None else 0
    if n_populations < 0:
        raise ValueError("kang_n_populations must be non-negative")
    n_features = int(kwargs.get("kang_n_features", 50))
    seed = int(kwargs.get("kang_seed", 0))
    group_keys = kwargs.get("kang_group_keys", ("Donor", "stim"))
    if isinstance(group_keys, str):
        group_keys = tuple(k.strip() for k in group_keys.split(",") if k.strip())

    cells_tag = str(cells_per_population) if cells_per_population else "all"
    populations_tag = str(n_populations) if n_populations else "all"
    cache_name = (
        f"populations_cells{cells_tag}_features{n_features}_"
        f"n{populations_tag}_seed{seed}_{'-'.join(group_keys)}.npz"
    )
    cache_path = os.path.join(root, cache_name)
    if os.path.isfile(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        PCs = [torch.from_numpy(arr.astype("float32")) for arr in cached["populations"]]
        labels = cached["labels"].astype(np.int64)
        num_labels = int(cached["num_labels"][0])
        return PCs, labels, num_labels

    adata = sc.read_h5ad(h5ad_path)
    missing = [key for key in group_keys if key not in adata.obs]
    if missing:
        raise ValueError(f"Kang PBMC data missing obs columns: {missing}")

    # Build a compact cell feature space with sparse operations. This avoids
    # densifying the full gene-by-cell matrix on login nodes.
    X = adata.X.astype(np.float32)
    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        min_value = float(X.data.min()) if X.nnz else 0.0
        if min_value >= 0:
            row_sums = np.asarray(X.sum(axis=1)).ravel()
            scale = np.divide(
                1e4,
                row_sums,
                out=np.zeros_like(row_sums, dtype=np.float32),
                where=row_sums > 0,
            )
            X = sparse.diags(scale).dot(X).tocsr()
            X.data = np.log1p(X.data)
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
    else:
        X = np.asarray(X, dtype=np.float32)
        min_value = float(np.min(X))
        if min_value >= 0:
            row_sums = X.sum(axis=1, keepdims=True)
            X = np.divide(
                X * 1e4,
                row_sums,
                out=np.zeros_like(X, dtype=np.float32),
                where=row_sums > 0,
            )
            X = np.log1p(X)
        mean = X.mean(axis=0)
        mean_sq = (X * X).mean(axis=0)
    var = mean_sq - mean * mean
    n_hvg = min(max(2000, n_features), X.shape[1])
    top_idx = np.argpartition(var, -n_hvg)[-n_hvg:]
    X_top = X[:, top_idx]
    svd = TruncatedSVD(n_components=n_features, random_state=seed)
    features = svd.fit_transform(X_top).astype("float32")
    features = StandardScaler().fit_transform(features).astype("float32")

    group_series = adata.obs[list(group_keys)].astype(str).agg("__".join, axis=1)
    group_names = sorted(group_series.unique())
    group_to_label = {name: i for i, name in enumerate(group_names)}
    group_indices = {
        name: np.flatnonzero(group_series.to_numpy() == name)
        for name in group_names
    }
    group_names = [name for name in group_names if len(group_indices[name]) > 0]

    rng = np.random.default_rng(seed)
    selected_groups = (
        group_names
        if n_populations == 0
        else [group_names[i % len(group_names)] for i in range(n_populations)]
    )
    populations = []
    labels = []
    population_group_names = []
    for group_name in selected_groups:
        idx = group_indices[group_name]
        if cells_per_population == 0:
            selected = idx
        else:
            selected = rng.choice(
                idx,
                size=cells_per_population,
                replace=len(idx) < cells_per_population,
            )
        populations.append(features[selected])
        labels.append(group_to_label[group_name])
        population_group_names.append(group_name)

    if cells_per_population == 0:
        populations_array = np.asarray(populations, dtype=object)
    else:
        populations_array = np.stack(populations).astype("float32")
    labels = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(
        cache_path,
        populations=populations_array,
        labels=labels,
        group_names=np.asarray(population_group_names, dtype=object),
        group_keys=np.asarray(group_keys, dtype=object),
        feature_columns=np.asarray([f"PCA_{i + 1}" for i in range(n_features)], dtype=object),
        population_sizes=np.asarray([len(pop) for pop in populations], dtype=np.int64),
        original_population_sizes=np.asarray(
            [len(group_indices[name]) for name in population_group_names], dtype=np.int64
        ),
        num_labels=np.asarray([len(group_names)], dtype=np.int64),
    )

    PCs = [torch.from_numpy(pop) for pop in populations]
    return PCs, labels, len(group_names)


def _load_population_npz(**kwargs):
    cache_path = kwargs.get("population_cache")
    if not cache_path:
        raise ValueError("--population_cache is required for --raw_dir population_npz")
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Population cache not found: {cache_path}")

    cached = np.load(cache_path, allow_pickle=True)
    if "populations" not in cached.files:
        raise ValueError("Population cache must contain a 'populations' array")
    PCs = [torch.from_numpy(np.asarray(arr, dtype=np.float32)) for arr in cached["populations"]]
    if not PCs or any(pc.ndim != 2 or pc.shape[0] == 0 for pc in PCs):
        raise ValueError("Every population must be a non-empty (n_points, n_features) array")
    feature_dims = {pc.shape[1] for pc in PCs}
    if len(feature_dims) != 1:
        raise ValueError(f"All populations must share one feature dimension; found {feature_dims}")

    labels = (
        np.asarray(cached["labels"], dtype=np.int64)
        if "labels" in cached.files
        else np.zeros(len(PCs), dtype=np.int64)
    )
    if len(labels) != len(PCs):
        raise ValueError("Population and label counts do not match")
    num_labels = int(cached["num_labels"][0]) if "num_labels" in cached.files else len(np.unique(labels))
    return PCs, labels, num_labels


def _load_pdo_caf_mendeley_populations(raw_dir, **kwargs):
    root = raw_dir
    if root in {"pdo_caf_mendeley", "pdo_caf"}:
        root = os.path.join("data", "pdo_caf_mendeley")
    cache_path = kwargs.get("pdo_caf_cache")
    if cache_path is None:
        cache_path = os.path.join(root, "populations_latest.npz")
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"PDO/CAF population cache not found: {cache_path}. "
            "Run preprocess_pdo_caf_mendeley.py first."
        )
    cached = np.load(cache_path, allow_pickle=True)
    PCs = [torch.from_numpy(np.asarray(arr, dtype=np.float32)) for arr in cached["populations"]]
    labels = cached["labels"].astype(np.int64)
    if "num_labels" in cached.files:
        num_labels = int(cached["num_labels"][0])
    else:
        num_labels = len(np.unique(labels))
    return PCs, labels, num_labels


def load_data(raw_dir, full, **kwargs):
    raw_dir = raw_dir.rstrip("/")
    data_name = raw_dir.split("/")[-1]
    if data_name == "melanoma_data_full":
        if full:
            suffix = "_full"
        else:
            suffix = ""
        with open(os.path.join(raw_dir, "pc" + suffix + ".pkl"), "rb") as handle:
            PCs = pickle.load(handle)
            PCs = [
                torch.tensor(StandardScaler().fit_transform(PCs[i]), dtype=torch.float)
                for i in range(len(PCs))
            ]
        labels = np.load(os.path.join(raw_dir, "labels" + suffix + ".npy"))
        num_labels = len(np.unique(labels))
    elif data_name == "COVID_data":
        with open(
            os.path.join(raw_dir, "filtered_point_clouds_final.pickle"), "rb"
        ) as handle:
            PCs = pickle.load(handle)
            PCs = [
                torch.tensor(
                    StandardScaler().fit_transform(PCs[i].values[:100]),
                    dtype=torch.float,
                )
                for i in PCs
            ]
        with open(
            os.path.join(raw_dir, "filtered_point_cloud_labels_final.pickle"), "rb"
        ) as handle:
            labels = list(pickle.load(handle).values())

        # with open(os.path.join(raw_dir, 'pc_covid.pkl'), 'rb') as handle:
        #     PCs = pickle.load(handle)
        #     PCs = [torch.tensor(StandardScaler().fit_transform(PCs[i]), dtype=torch.float) for i in range(len(PCs))]
        # with open(os.path.join(raw_dir, 'patient_list_covid.pkl'), 'rb') as handle:
        #     subsampled_patient_ids = pickle.load(handle)
        # labels = np.load(os.path.join(raw_dir, 'labels.npy'))
        num_labels = len(np.unique(labels))
    elif data_name == "pdo_data":
        with open(osp.join(raw_dir, "pc_pdo_treatment.pkl"), "rb") as handle:
            PCs = pickle.load(handle)
            PCs = [
                torch.tensor(StandardScaler().fit_transform(PCs[i]), dtype=torch.float)
                for i in range(len(PCs))
            ]
        keep = []
        for i in range(len(PCs)):
            if PCs[i].shape[0] > 200:
                keep.append(i)
        le = LabelEncoder()
        labels = le.fit_transform(
            np.load(osp.join(raw_dir, "labels_pdo_treatment.npy"))
        )
        PCs = [PCs[i] for i in keep]
        labels = [labels[i] for i in keep]
        num_labels = len(np.unique(labels))
    elif data_name == "sea":
        X = []
        X_spatial = []
        labels = []
        for file in pathlib.Path(raw_dir + "_matrix/").glob("*.pt"):
            X.append(torch.load(file))
        for file in pathlib.Path(raw_dir + "_spatial/").glob("*.pt"):
            X_spatial.append(torch.load(file))
        for file in pathlib.Path(raw_dir + "_labels/").glob("*.pt"):
            labels.append(torch.load(file))
        return X, X_spatial, labels
    elif data_name == "synthetic_gmm":
        PCs = torch.load(raw_dir + "/point_clouds.pt")
        labels = torch.load(raw_dir + "/labels.pt")[:, 0].tolist()
        num_labels = 0
    elif data_name in {"modelnet40", "ModelNet40", "modelnet40_hdf5"}:
        import h5py
        import urllib.request

        root = kwargs.get("modelnet_root") or raw_dir
        if root in {"modelnet40", "ModelNet40", "modelnet40_hdf5"}:
            root = os.path.join("data", "modelnet40_hdf5")
        os.makedirs(root, exist_ok=True)

        dataset_dir = os.path.join(root, "modelnet40_ply_hdf5_2048")
        os.makedirs(dataset_dir, exist_ok=True)

        num_points = int(kwargs.get("modelnet_num_points", 2048))
        if num_points < 1 or num_points > 2048:
            raise ValueError("modelnet_num_points must be between 1 and 2048")
        max_samples = kwargs.get("max_samples")
        max_samples = None if max_samples is None else int(max_samples)

        mirror = "https://share.phys.ethz.ch/~gseg/Predator/data/modelnet40_ply_hdf5_2048"
        h5_names = [f"ply_data_train{i}.h5" for i in range(5)]

        PCs = []
        labels = []
        for name in tqdm(h5_names, desc="Loading ModelNet40 HDF5"):
            h5_path = os.path.join(dataset_dir, name)
            if not os.path.isfile(h5_path):
                url = f"{mirror}/{name}"
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, h5_path)
            with h5py.File(h5_path, "r") as handle:
                data = handle["data"][:, :num_points, :].astype("float32")
                label = handle["label"][:, 0].astype("int64")
            for pc, y in zip(data, label):
                PCs.append(torch.from_numpy(pc))
                labels.append(int(y))
                if max_samples is not None and len(PCs) >= max_samples:
                    break
            if max_samples is not None and len(PCs) >= max_samples:
                break

        labels = np.asarray(labels, dtype=np.int64)
        num_labels = 40
    elif data_name in {"kang_pbmc", "kang_ifnb", "pbmc_ifnb"}:
        PCs, labels, num_labels = _load_kang_pbmc_populations(raw_dir, **kwargs)
    elif data_name in {"pdo_caf_mendeley", "pdo_caf"}:
        PCs, labels, num_labels = _load_pdo_caf_mendeley_populations(raw_dir, **kwargs)
    elif data_name == "population_npz":
        PCs, labels, num_labels = _load_population_npz(**kwargs)
    else:
        raise ValueError(f"Dataset {data_name} not recognized.")
    return PCs, labels, num_labels


def load_data_persistence(raw_dir, full):
    data = np.load(os.path.join(raw_dir, "pc_persistence.npy"), allow_pickle=True)

    PCs = [torch.tensor(i["pc"], dtype=torch.float) for i in data]
    h0 = torch.from_numpy(np.vstack([i["h0_bc"] for i in data]))
    h1 = torch.from_numpy(np.vstack([i["h1_bc"][:99] for i in data]))
    labels = F.normalize(torch.cat([h0, h1], 1))
    return PCs, labels, labels.shape[1]


def load_data_ST(raw_dir: str, dataset: str, label_name: str):
    raw_dir = raw_dir.rstrip("/")
    spatial_cords = torch.load(f"{raw_dir}/spatial_cords_{dataset}_{label_name}.pt")
    num_pcs = len(spatial_cords)
    gene_expr = torch.load(f"{raw_dir}/gene_expr_{dataset}_{label_name}.pt")
    labels = torch.load(f"{raw_dir}/labels_{dataset}_{label_name}.pt")
    indices = torch.load(f"{raw_dir}/indices_{dataset}_{label_name}.pt")
    spatial_cords = [spatial_cords[i][indices[i]].float() for i in range(num_pcs)]
    gene_expr = [gene_expr[i][indices[i]].float() for i in range(num_pcs)]

    return spatial_cords, gene_expr, labels, 2


def load_data_ST_melanoma(root):
    graph_names = [f[:-3] for f in os.listdir(root) if f.endswith(".pt")]
    graph_files = [f for f in os.listdir(root) if f.endswith(".pt")]
    patient_metadata = pd.read_csv(
        "/gpfs/gibbs/pi/krishnaswamy_smita/hm638/SCGFM/data/Melanoma/patient_info.csv"
    )
    max_num_features = 29
    spatial_cords = []
    gene_expr = []
    labels = []

    print("Preprocessing data!")
    for idx in tqdm(range(len(graph_names))):
        label = patient_metadata[
            patient_metadata.id == graph_names[idx]
        ].response_binary.values[0]
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
