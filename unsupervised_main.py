import gc
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
from models.graph_learning import HiPoNet, HiPoNetAutoencoder

gc.enable()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = ArgumentParser(description="HiPoNet unsupervised autoencoder with UDEMD regularisation")
parser.add_argument("--raw_dir", type=str, default="pdo_data")
parser.add_argument("--full", action="store_true")
parser.add_argument("--num_weights", type=int, default=2)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--K", type=int, default=1, help="Order of simplicial complex")
parser.add_argument("--J", type=int, default=3)
parser.add_argument("--latent_dim", type=int, default=64, help="Latent space dimensionality")
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


def make_collate_fn(udemd_dists: np.ndarray | None, max_points: int):
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

        input_tensor = torch.nested.as_nested_tensor(
            [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
             for x in pcs],
            layout=torch.jagged,
        ).to_padded_tensor(padding=0.0)

        # Force a fixed number of points so decoder targets are shape-stable.
        cur_n = input_tensor.shape[1]
        if cur_n < max_points:
            pad = torch.zeros(
                input_tensor.shape[0],
                max_points - cur_n,
                input_tensor.shape[2],
                dtype=input_tensor.dtype,
            )
            input_tensor = torch.cat([input_tensor, pad], dim=1)
        elif cur_n > max_points:
            input_tensor = input_tensor[:, :max_points]

        mask = input_tensor.sum(-1) != 0

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

def run_epoch(model, loader, opt, accumulate_steps, device, train=True):
    model.train(train)
    total_loss = total_recon = total_dist = 0.0
    n_batches = len(loader)
    if opt is not None:
        opt.zero_grad()

    for i, (batch, mask, target_dists) in enumerate(loader, start=1):
        batch = batch.to(device)
        mask = mask.to(device)
        if target_dists is not None:
            target_dists = target_dists.to(device)

        loss, recon_loss, dist_loss = model.compute_loss(
            batch, mask, target_dists=target_dists
        )

        if train:
            (loss / accumulate_steps).backward()
            if (i % accumulate_steps == 0) or i == n_batches:
                opt.step()
                opt.zero_grad()

        total_loss += loss.detach().item()
        total_recon += recon_loss.detach().item()
        total_dist += dist_loss.detach().item()

        del batch, mask, loss, recon_loss, dist_loss
        if target_dists is not None:
            del target_dists
        torch.cuda.empty_cache()
        gc.collect()

    return total_loss / n_batches, total_recon / n_batches, total_dist / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = vars(args)
    config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")

    # Load data — labels are ignored for unsupervised training
    PCs, labels, _ = load_data(args.raw_dir, args.full)

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
    split = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, list(range(split)))
    val_set   = torch.utils.data.Subset(dataset, list(range(split, len(dataset))))

    collate = make_collate_fn(udemd_dists, max_points=max_points)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
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
    ).to(args.device)

    # Infer wavelet feature dimension with a dummy forward pass
    with torch.no_grad():
        dummy_pc = PCs[0].unsqueeze(0).to(args.device)
        dummy_mask = torch.ones((1, PCs[0].shape[0]), dtype=torch.bool, device=args.device)
        wavelet_dim = hiponet(dummy_pc, dummy_mask).shape[1]
    print(f"Wavelet feature dim: {wavelet_dim}  |  Latent dim: {args.latent_dim}")

    model = HiPoNetAutoencoder(
        hiponet=hiponet,
        wavelet_dim=wavelet_dim,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        point_dim=input_dim,
        max_points=max_points,
        dist_weight=args.dist_weight,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    with tqdm(range(1, args.num_epochs + 1)) as tq:
        for epoch in tq:
            tr_loss, tr_recon, tr_dist = run_epoch(
                model, train_loader, opt,
                accumulate_steps=args.n_accumulate,
                device=args.device,
                train=True,
            )
            with torch.no_grad():
                val_loss, val_recon, val_dist = run_epoch(
                    model, val_loader, opt=None,
                    accumulate_steps=1,
                    device=args.device,
                    train=False,
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
                    "val/loss": val_loss,
                    "val/recon_loss": val_recon,
                    "val/dist_loss": val_dist,
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
                f"Epoch {epoch:3d}  tr={tr_loss:.4f}  (recon={tr_recon:.4f} dist={tr_dist:.4f})  "
                f"val={val_loss:.4f}  best={best_val_loss:.4f}"
            )

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {os.path.join(args.save_dir, 'best_model.pt')}")

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
            z, _, _ = model(batch, mask)
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


if __name__ == "__main__":
    main()
