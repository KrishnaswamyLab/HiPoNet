import gc
import json
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from tqdm import tqdm

import wandb

from utils.read_data import load_data
from models.graph_learning import HiPoNet, HiPoNetAutoencoder, HiPoNetVAE, HiPoNetWaveletAutoencoder

gc.enable()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = ArgumentParser(description="HiPoNet unsupervised autoencoder with UDEMD regularisation")
parser.add_argument("--raw_dir", type=str, default="pdo_data")
parser.add_argument("--full", action="store_true")
parser.add_argument(
    "--modelnet_num_points",
    type=int,
    default=2048,
    help="Number of points to sample per ModelNet40 shape when --raw_dir modelnet40",
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Optional cap on loaded point clouds for smoke tests/debugging",
)
parser.add_argument(
    "--train_max_samples",
    type=int,
    default=None,
    help="Optional deterministic cap on the training split; all loaded samples are still exported",
)
parser.add_argument(
    "--val_max_samples",
    type=int,
    default=None,
    help="Optional deterministic cap on the validation split; all loaded samples are still exported",
)
parser.add_argument(
    "--kang_cells_per_population",
    type=int,
    default=128,
    help="Cells sampled per Kang population; use 0 to retain every cell",
)
parser.add_argument(
    "--kang_n_populations",
    type=int,
    default=256,
    help="Number of Kang populations; use 0 for each unique metadata group once",
)
parser.add_argument(
    "--kang_n_features",
    type=int,
    default=50,
    help="Number of PCA expression features per cell for --raw_dir kang_pbmc",
)
parser.add_argument(
    "--kang_group_keys",
    type=str,
    default="Donor,stim",
    help="Comma-separated AnnData obs columns defining biological populations",
)
parser.add_argument(
    "--kang_seed",
    type=int,
    default=0,
    help="Random seed for sampling Kang PBMC populations",
)
parser.add_argument(
    "--pdo_caf_cache",
    type=str,
    default=None,
    help="Path to a preprocessed PDO/CAF all-cells population cache.",
)
parser.add_argument(
    "--population_cache",
    type=str,
    default=None,
    help="Generic NPZ cache for --raw_dir population_npz; must contain populations",
)
parser.add_argument(
    "--split_cache",
    type=str,
    default=None,
    help="Population NPZ containing group_names/group_keys for metadata-held-out splitting",
)
parser.add_argument("--split_field", type=str, default=None)
parser.add_argument("--val_group", type=str, default=None)
parser.add_argument("--test_group", type=str, default=None)
parser.add_argument("--num_weights", type=int, default=2)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--K", type=int, default=1, help="Order of simplicial complex")
parser.add_argument(
    "--use_geometric_laplacian",
    action="store_true",
    help="Use the metric-aware geometric Hodge Laplacian for K >= 2",
)
parser.add_argument("--J", type=int, default=3)
parser.add_argument("--latent_dim", type=int, default=64, help="Latent space dimensionality")
parser.add_argument("--vae", action="store_true", help="Train a variational autoencoder")
parser.add_argument(
    "--autoencoder_target",
    choices=["points", "wavelet"],
    default="points",
    help="Train the bottleneck to reconstruct original points or HiPoNet wavelet features.",
)
parser.add_argument(
    "--recon_loss",
    type=str,
    choices=["chamfer", "mse"],
    default="chamfer",
    help="Point-cloud reconstruction loss. Chamfer is order-invariant and is the VAE default.",
)
parser.add_argument(
    "--generator_weight",
    type=float,
    default=0.0,
    help=(
        "For --autoencoder_target wavelet, train a separate detached latent-to-points "
        "generation head with this loss weight."
    ),
)
parser.add_argument(
    "--kl_weight",
    type=float,
    default=1e-3,
    help="Final beta weight for the VAE KL loss",
)
parser.add_argument(
    "--kl_warmup_epochs",
    type=int,
    default=10,
    help="Linearly warm KL weight from 0 to --kl_weight over this many epochs",
)
parser.add_argument(
    "--n_generated_samples",
    type=int,
    default=16,
    help="Number of prior samples to decode and save after VAE training",
)
parser.add_argument(
    "--hidden_dims",
    type=int,
    nargs="+",
    default=[256, 128],
    help="Hidden layer widths for encoder/decoder MLPs",
)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=3e-3)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--split_seed", type=int, default=0)
parser.add_argument("--val_fraction", type=float, default=0.2)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--disable_wb", action="store_true")
parser.add_argument(
    "--n_accumulate",
    type=int,
    default=1,
    help="Gradient accumulation steps",
)
# UDEMD distance preservation loss
parser.add_argument(
    "--dist_weight",
    type=float,
    default=0.1,
    help="Weight λ for the UDEMD distance preservation loss (0 to disable)",
)
parser.add_argument(
    "--udemd_knn",
    type=int,
    default=10,
    help="Number of nearest neighbours for the global k-NN graph used by UDEMD",
)
parser.add_argument(
    "--udemd_n_scales",
    type=int,
    default=6,
    help="Number of dyadic diffusion scales for UDEMD (scales = 1, 2, 4, …, 2^(n_scales-1))",
)
parser.add_argument(
    "--udemd_sigma",
    type=float,
    default=None,
    help="Gaussian bandwidth for the global UDEMD graph (auto-tuned if None)",
)
parser.add_argument(
    "--udemd_cache",
    type=str,
    default=None,
    help="Path to cache / load the precomputed UDEMD distance matrix (.npy). "
         "If the file exists it is loaded; otherwise it is computed and saved.",
)
parser.add_argument(
    "--decoded_dist_weight",
    type=float,
    default=0.0,
    help="Weight for preserving pairwise decoded-population Chamfer distances. "
         "Only applies to the deterministic HiPoNet autoencoder.",
)
# Alpha regularisation (kept from supervised script for consistency)
parser.add_argument("--normalize_alphas", action="store_true")
parser.add_argument("--ignore_alphas", action="store_true")
parser.add_argument("--softmax_alphas", action="store_true")
parser.add_argument("--use_alphas_for_connectivity_only", action="store_true")
parser.add_argument("--orthogonal", action="store_true")
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--sparse_lambda", type=float, default=0.01)
# Checkpoint
parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save model checkpoints")
args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = "cuda"
else:
    args.device = "cpu"


# ---------------------------------------------------------------------------
# UDEMD pre-computation
# ---------------------------------------------------------------------------

def compute_udemd_distances(
    PCs: list,
    knn: int = 10,
    n_scales: int = 6,
    sigma: float | None = None,
) -> np.ndarray:
    """Compute pairwise Unbalanced Diffusion Earth Mover's Distances between
    point clouds.

    Each point cloud is treated as a uniform probability distribution over its
    cells.  A shared k-NN graph is built over all cells from all samples; the
    diffusion operator P is computed on this graph.  For each sample i the
    distribution μ_i places mass 1/N_i on its cells and zero elsewhere.

    The UDEMD between samples i and j is approximated by the L1 distance
    between their stacked diffusion embeddings::

        embed_i = concat(w_t * P^t μ_i  for t in [1, 2, 4, ..., 2^(n_scales-1)])
        UDEMD(i, j) = ||embed_i - embed_j||_1

    Parameters
    ----------
    PCs : list of torch.Tensor, each of shape (N_i, d)
    knn : int
        Number of nearest neighbours for the global graph.
    n_scales : int
        Number of dyadic diffusion scales.
    sigma : float or None
        Gaussian bandwidth.  Auto-tuned from the median squared
        nearest-neighbour distance if None.

    Returns
    -------
    distances : np.ndarray, shape (K, K)
        Symmetric pairwise UDEMD distance matrix.
    """
    K = len(PCs)

    all_cells_list = [pc.detach().cpu().numpy().astype(np.float32) for pc in PCs]
    sample_sizes = [arr.shape[0] for arr in all_cells_list]
    all_cells = np.concatenate(all_cells_list, axis=0)   # (N_total, d)
    N_total = all_cells.shape[0]
    sample_ids = np.repeat(np.arange(K), sample_sizes)  # (N_total,)

    print(f"  UDEMD: {K} samples, {N_total} total cells, {all_cells.shape[1]} features")

    nn_model = NearestNeighbors(n_neighbors=knn + 1, metric="euclidean", n_jobs=-1)
    nn_model.fit(all_cells)
    nn_dists, nn_idx = nn_model.kneighbors(all_cells)  # (N_total, knn+1)

    if sigma is None:
        sigma = float(np.median(nn_dists[:, 1] ** 2))
        sigma = max(sigma, 1e-8)
        print(f"  UDEMD: auto sigma = {sigma:.4g}")

    # Build sparse symmetric Gaussian-weighted adjacency (no self-loops)
    rows, cols, vals = [], [], []
    for i in range(N_total):
        for nb in range(1, knn + 1):
            j = nn_idx[i, nb]
            w = float(np.exp(-(nn_dists[i, nb] ** 2) / sigma))
            rows.append(i)
            cols.append(j)
            vals.append(w)
    W_sp = sp.csr_matrix((vals, (rows, cols)), shape=(N_total, N_total), dtype=np.float32)
    W_sp = (W_sp + W_sp.T).multiply(0.5)  # symmetrize

    # Row-normalise to get Markov matrix P
    row_sums = np.array(W_sp.sum(axis=1)).flatten()
    row_sums = np.maximum(row_sums, 1e-8)
    P = sp.diags(1.0 / row_sums) @ W_sp  # (N_total, N_total) sparse
    P = P.astype(np.float32)

    # Per-sample uniform distributions: distrib[:, k] = 1/N_k for sample k's cells
    distrib = np.zeros((N_total, K), dtype=np.float32)
    for k in range(K):
        idx = sample_ids == k
        n_k = int(idx.sum())
        if n_k > 0:
            distrib[idx, k] = 1.0 / n_k

    # Accumulate pairwise L1 distances across dyadic diffusion scales
    scales = [2 ** i for i in range(n_scales)]
    dists = np.zeros((K, K), dtype=np.float64)
    D_current = distrib.copy()   # (N_total, K)
    current_t = 0

    for t_idx, target_t in enumerate(tqdm(scales, desc="  UDEMD scales", leave=False)):
        for _ in range(target_t - current_t):
            D_current = P @ D_current   # sparse × dense
        current_t = target_t

        weight = 0.5 ** (n_scales - t_idx - 1)
        # Pairwise L1 distances between diffused distributions at this scale
        scale_dists = cdist(D_current.T, D_current.T, metric="cityblock")  # (K, K)
        dists += weight * scale_dists

    dists = 0.5 * (dists + dists.T)  # enforce symmetry numerically
    return dists.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset / Collate
# ---------------------------------------------------------------------------

class PointCloudDataset(torch.utils.data.Dataset):
    """Wraps a list of point clouds, returning (pc, original_index) per item."""

    def __init__(self, PCs: list):
        self.PCs = PCs

    def __len__(self) -> int:
        return len(self.PCs)

    def __getitem__(self, idx: int):
        return self.PCs[idx], idx


def make_collate_fn(udemd_dists: np.ndarray | None, pad_to_points: int | None = None):
    """Return a collate function that pads point clouds and extracts the
    upper-triangular UDEMD sub-matrix for the batch.

    Parameters
    ----------
    udemd_dists : (K, K) numpy array or None
        Full pairwise UDEMD distance matrix over all samples.
        Pass None when ``dist_weight == 0``.
    """
    def collate(batch):
        pcs, indices = zip(*batch)
        indices = list(indices)
        lengths = [pc.shape[0] for pc in pcs]

        input_tensor = torch.nested.as_nested_tensor(
            [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
             for x in pcs],
            layout=torch.jagged,
        ).to_padded_tensor(padding=0.0)

        # Point reconstruction needs a stable target shape. Wavelet-only
        # training leaves this unset and pads only to the batch maximum.
        cur_n = input_tensor.shape[1]
        if pad_to_points is not None and cur_n < pad_to_points:
            pad = torch.zeros(
                input_tensor.shape[0],
                pad_to_points - cur_n,
                input_tensor.shape[2],
                dtype=input_tensor.dtype,
            )
            input_tensor = torch.cat([input_tensor, pad], dim=1)
        elif pad_to_points is not None and cur_n > pad_to_points:
            input_tensor = input_tensor[:, :pad_to_points]
            lengths = [min(length, pad_to_points) for length in lengths]

        arange = torch.arange(input_tensor.shape[1])
        mask = arange.unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)

        if udemd_dists is not None:
            B = len(indices)
            sub_mat = udemd_dists[np.ix_(indices, indices)]  # (B, B)
            ii, jj = np.triu_indices(B, k=1)
            target_dists = torch.tensor(sub_mat[ii, jj], dtype=torch.float32)
        else:
            target_dists = None

        return input_tensor, mask, target_dists

    return collate


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    loader,
    opt,
    accumulate_steps,
    device,
    train=True,
    kl_weight: float | None = None,
):
    model.train(train)
    total_loss = total_recon = total_dist = total_decoded_dist = total_kl = 0.0
    n_batches = len(loader)
    if opt is not None:
        opt.zero_grad()

    for i, (batch, mask, target_dists) in enumerate(loader, start=1):
        batch = batch.to(device)
        mask = mask.to(device)
        if target_dists is not None:
            target_dists = target_dists.to(device)

        if kl_weight is None:
            losses = model.compute_loss(batch, mask, target_dists=target_dists)
        else:
            losses = model.compute_loss(
                batch, mask, target_dists=target_dists, kl_weight=kl_weight
            )
        if len(losses) == 3:
            loss, recon_loss, dist_loss = losses
            decoded_dist_loss = batch.new_zeros(())
            kl_loss = batch.new_zeros(())
        elif len(losses) == 4:
            loss, recon_loss, dist_loss, kl_loss = losses
            decoded_dist_loss = batch.new_zeros(())
        else:
            loss, recon_loss, dist_loss, decoded_dist_loss, kl_loss = losses

        if train:
            (loss / accumulate_steps).backward()
            if (i % accumulate_steps == 0) or i == n_batches:
                opt.step()
                opt.zero_grad()

        total_loss += loss.detach().item()
        total_recon += recon_loss.detach().item()
        total_dist += dist_loss.detach().item()
        total_decoded_dist += decoded_dist_loss.detach().item()
        total_kl += kl_loss.detach().item()

        del batch, mask, loss, recon_loss, dist_loss, decoded_dist_loss, kl_loss
        if target_dists is not None:
            del target_dists
        torch.cuda.empty_cache()
        gc.collect()

    return (
        total_loss / n_batches,
        total_recon / n_batches,
        total_dist / n_batches,
        total_decoded_dist / n_batches,
        total_kl / n_batches,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = vars(args)
    config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")

    # Load data — labels are ignored for unsupervised training except for saved diagnostics.
    PCs, labels, _ = load_data(
        args.raw_dir,
        args.full,
        modelnet_num_points=args.modelnet_num_points,
        max_samples=args.max_samples,
        kang_cells_per_population=args.kang_cells_per_population,
        kang_n_populations=args.kang_n_populations,
        kang_n_features=args.kang_n_features,
        kang_group_keys=args.kang_group_keys,
        kang_seed=args.kang_seed,
        pdo_caf_cache=args.pdo_caf_cache,
        population_cache=args.population_cache,
    )

    if os.environ.get("SMOKE_TEST"):
        PCs = [pc[:60 + i] for i, pc in enumerate(PCs[:32])]
        args.disable_wb = True

    wandb.init(
        project="hiponet-unsupervised",
        config=config,
        mode="disabled" if args.disable_wb else None,
    )

    # ------------------------------------------------------------------
    # Pre-compute UDEMD pairwise distances (once, before training)
    # ------------------------------------------------------------------
    udemd_dists = None
    if args.dist_weight > 0.0:
        if args.udemd_cache and os.path.isfile(args.udemd_cache):
            print(f"Loading cached UDEMD distances from {args.udemd_cache}")
            udemd_dists = np.load(args.udemd_cache)
        else:
            print("Computing UDEMD pairwise distances ...")
            udemd_dists = compute_udemd_distances(
                PCs,
                knn=args.udemd_knn,
                n_scales=args.udemd_n_scales,
                sigma=args.udemd_sigma,
            )
            if args.udemd_cache:
                os.makedirs(os.path.dirname(os.path.abspath(args.udemd_cache)), exist_ok=True)
                np.save(args.udemd_cache, udemd_dists)
                print(f"UDEMD distances saved to {args.udemd_cache}")

    # ------------------------------------------------------------------
    # Dataset / dataloaders
    # ------------------------------------------------------------------
    max_points = max(pc.shape[0] for pc in PCs)
    dataset = PointCloudDataset(PCs)
    test_indices = []
    if args.split_field is not None:
        split_cache_path = args.split_cache or args.population_cache
        if not split_cache_path:
            raise ValueError("--split_cache is required with --split_field")
        split_cache = np.load(split_cache_path, allow_pickle=True)
        group_keys = [str(value) for value in split_cache["group_keys"]]
        matches = [
            index for index, key in enumerate(group_keys)
            if key.lower() == args.split_field.lower()
        ]
        if not matches:
            raise ValueError(f"Split field {args.split_field!r} not found in {group_keys}")
        field_index = matches[0]
        split_values = np.asarray(
            [str(name).split("__")[field_index] for name in split_cache["group_names"]]
        )
        if len(split_values) != len(dataset):
            raise ValueError("Split metadata and population counts do not match")
        if args.val_group is None:
            raise ValueError("--val_group is required with --split_field")
        val_mask = split_values == str(args.val_group)
        test_mask = (
            split_values == str(args.test_group)
            if args.test_group is not None
            else np.zeros(len(dataset), dtype=bool)
        )
        train_indices = np.flatnonzero(~val_mask & ~test_mask).tolist()
        val_indices = np.flatnonzero(val_mask).tolist()
        test_indices = np.flatnonzero(test_mask).tolist()
        if not train_indices or not val_indices:
            raise ValueError("Metadata split produced an empty training or validation set")
    else:
        if not 0.0 < args.val_fraction < 1.0:
            raise ValueError("--val_fraction must be between 0 and 1")
        indices = np.random.default_rng(args.split_seed).permutation(len(dataset))
        split = int(round((1.0 - args.val_fraction) * len(dataset)))
        split = min(max(split, 1), len(dataset) - 1)
        train_indices = indices[:split].tolist()
        val_indices = indices[split:].tolist()
    subset_rng = np.random.default_rng(args.split_seed)
    if args.train_max_samples is not None and len(train_indices) > args.train_max_samples:
        train_indices = sorted(
            subset_rng.choice(train_indices, size=args.train_max_samples, replace=False).tolist()
        )
    if args.val_max_samples is not None and len(val_indices) > args.val_max_samples:
        val_indices = sorted(
            subset_rng.choice(val_indices, size=args.val_max_samples, replace=False).tolist()
        )
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    pad_to_points = max_points if args.autoencoder_target == "points" else None
    collate = make_collate_fn(udemd_dists, pad_to_points=pad_to_points)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    input_dim = PCs[0].shape[1]

    # Build HiPoNet backbone (pooling=True required for the autoencoder)
    hiponet = HiPoNet(
        dimension=input_dim,
        n_weights=args.num_weights,
        threshold=args.threshold,
        K=args.K,
        J=args.J,
        device=args.device,
        sigma=args.sigma,
        pooling=True,
        normalize_alphas=args.normalize_alphas,
        ignore_alphas=args.ignore_alphas,
        use_alphas_for_connectivity_only=args.use_alphas_for_connectivity_only,
        softmax_alphas=args.softmax_alphas,
        use_geometric_laplacian=args.use_geometric_laplacian,
    ).to(args.device)

    # Infer wavelet feature dimension with a dummy forward pass
    with torch.no_grad():
        dummy_pc = PCs[0].unsqueeze(0).to(args.device)
        dummy_mask = torch.ones((1, PCs[0].shape[0]), dtype=torch.bool, device=args.device)
        wavelet_dim = hiponet(dummy_pc, dummy_mask).shape[1]
    print(f"Wavelet feature dim: {wavelet_dim}  |  Latent dim: {args.latent_dim}")

    if args.vae and args.autoencoder_target == "wavelet":
        raise ValueError("--vae is only implemented for --autoencoder_target points")

    if args.autoencoder_target == "wavelet":
        autoencoder_cls = HiPoNetWaveletAutoencoder
    else:
        autoencoder_cls = HiPoNetVAE if args.vae else HiPoNetAutoencoder
    model_kwargs = {
        "hiponet": hiponet,
        "wavelet_dim": wavelet_dim,
        "latent_dim": args.latent_dim,
        "hidden_dims": args.hidden_dims,
        "point_dim": input_dim,
        "max_points": max_points,
        "dist_weight": args.dist_weight,
        "recon_loss_type": args.recon_loss,
    }
    if args.autoencoder_target == "wavelet":
        model_kwargs["generator_weight"] = args.generator_weight
    elif not args.vae:
        model_kwargs["decoded_dist_weight"] = args.decoded_dist_weight
    if args.vae:
        model_kwargs["kl_weight"] = args.kl_weight
    model = autoencoder_cls(**model_kwargs).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    with tqdm(range(1, args.num_epochs + 1)) as tq:
        for epoch in tq:
            if args.vae and args.kl_warmup_epochs > 0:
                kl_weight = args.kl_weight * min(1.0, epoch / args.kl_warmup_epochs)
            else:
                kl_weight = args.kl_weight if args.vae else None

            tr_loss, tr_recon, tr_dist, tr_decoded_dist, tr_kl = run_epoch(
                model, train_loader, opt,
                accumulate_steps=args.n_accumulate,
                device=args.device,
                train=True,
                kl_weight=kl_weight,
            )
            with torch.no_grad():
                val_loss, val_recon, val_dist, val_decoded_dist, val_kl = run_epoch(
                    model, val_loader, opt=None,
                    accumulate_steps=1,
                    device=args.device,
                    train=False,
                    kl_weight=kl_weight,
                )
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

            # Log alpha values
            if hasattr(model.hiponet.layer, "alphas"):
                for k, row in enumerate(model.hiponet.layer.alphas):
                    for d, v in enumerate(row):
                        wandb.log({f"Alpha{k}_{d}": v.item()}, step=epoch)

            # Gradient norms
            for name, param in model.named_parameters():
                if param.grad is not None:
                    wandb.log({f"{name}.grad_norm": param.grad.norm().item()}, step=epoch)

            wandb.log(
                {
                    "train/loss": tr_loss,
                    "train/recon_loss": tr_recon,
                    "train/dist_loss": tr_dist,
                    "train/decoded_dist_loss": tr_decoded_dist,
                    "train/kl_loss": tr_kl,
                    "val/loss": val_loss,
                    "val/recon_loss": val_recon,
                    "val/dist_loss": val_dist,
                    "val/decoded_dist_loss": val_decoded_dist,
                    "val/kl_loss": val_kl,
                    "kl_weight": kl_weight or 0.0,
                    "best_val_loss": best_val_loss,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            if args.orthogonal and hasattr(model.hiponet.layer, "alphas"):
                alphas = model.hiponet.layer.alphas
                orth_loss = (
                    (alphas @ alphas.T - torch.eye(args.num_weights, device=args.device))
                    .square()
                    .mean()
                )
                wandb.log({"train/orth_loss": orth_loss.item()}, step=epoch)

            tq.set_description(
                f"Epoch {epoch:3d}  tr={tr_loss:.4f}  "
                f"(recon={tr_recon:.4f} dist={tr_dist:.4f} decdist={tr_decoded_dist:.4f} kl={tr_kl:.4f})  "
                f"val={val_loss:.4f}  best={best_val_loss:.4f}"
            )

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    checkpoint_path = os.path.join(args.save_dir, "best_model.pt")
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Export embeddings from the model selected by validation loss, not the
    # final in-memory epoch.
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    print("Reloaded best checkpoint for latent export.")
    with open(os.path.join(args.save_dir, "training_summary.json"), "w") as handle:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices,
                "split_field": args.split_field,
                "val_group": args.val_group,
                "test_group": args.test_group,
                "batch_size": args.batch_size,
                "n_loaded_samples": len(dataset),
                "train_max_samples": args.train_max_samples,
                "val_max_samples": args.val_max_samples,
                "dist_weight": args.dist_weight,
                "autoencoder_target": args.autoencoder_target,
                "latent_dim": args.latent_dim,
                "seed": args.seed,
                "split_seed": args.split_seed,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    # Save final latent representations for all data
    print("Computing final latent representations...")
    all_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch, mask, _ in all_loader:
            batch, mask = batch.to(args.device), mask.to(args.device)
            z, _ = model.encode(batch, mask)
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0).numpy()
    latent_path = os.path.join(args.save_dir, "latent_representations.npy")
    np.save(latent_path, all_z)
    # Backward-compatible filename used previously.
    np.save(os.path.join(args.save_dir, "embeddings.npy"), all_z)

    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    labels_path = os.path.join(args.save_dir, "labels.npy")
    np.save(labels_path, labels_np)

    print(f"Latent representations saved to: {latent_path}")
    print(f"Labels saved to: {labels_path}")

    if args.vae and args.n_generated_samples > 0:
        with torch.no_grad():
            generated = model.sample(args.n_generated_samples, device=args.device).cpu().numpy()
        generated_path = os.path.join(args.save_dir, "generated_point_clouds.npy")
        np.save(generated_path, generated)
        print(f"Generated point clouds saved to: {generated_path}")


if __name__ == "__main__":
    main()
