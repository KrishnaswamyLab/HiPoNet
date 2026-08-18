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
from models.graph_learning import HiPoNet, HiPoNetAutoencoder

gc.enable()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = ArgumentParser(description="HiPoNet unsupervised autoencoder with phenoGS regularisation")
parser.add_argument("--raw_dir", type=str, default="pdo_data")
parser.add_argument("--full", action="store_true")
parser.add_argument(
    "--population_cache",
    type=str,
    default=None,
    help="NPZ cache containing Mendeley point-cloud populations and metadata",
)
parser.add_argument(
    "--pdo_caf_cache",
    type=str,
    default=None,
    help="Explicit Mendeley PDO/CAF population-cache path",
)
parser.add_argument(
    "--split_cache",
    type=str,
    default=None,
    help="Population NPZ containing group_names/group_keys for held-out splitting",
)
parser.add_argument("--split_field", type=str, default=None)
parser.add_argument("--val_group", type=str, default=None)
parser.add_argument("--test_group", type=str, default=None)
parser.add_argument("--train_max_samples", type=int, default=None)
parser.add_argument("--val_max_samples", type=int, default=None)
parser.add_argument("--num_weights", type=int, default=2)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--K", type=int, default=1, help="Order of simplicial complex")
parser.add_argument(
    "--use_geometric_laplacian",
    action="store_true",
    help="Use the metric-aware geometric Hodge Laplacian for K >= 2",
)
parser.add_argument(
    "--simplicial_max_neighbors",
    type=int,
    default=None,
    help="Mutual-neighbor cap for sparse K>=2 Gaussian graph construction",
)
parser.add_argument(
    "--simplicial_max_triangles",
    type=int,
    default=250,
    help="Maximum number of triangles retained per point cloud and view",
)
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
# phenoGS distance preservation loss
parser.add_argument(
    "--dist_weight",
    type=float,
    default=0.1,
    help="Weight λ for the phenoGS distance preservation loss (0 to disable)",
)
parser.add_argument(
    "--phenoGS_cache",
    type=str,
    default=None,
    help="Path to cache / load the precomputed phenoGS distance matrix (.npy). "
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
parser.add_argument("--skip_phate", action="store_true", help="Do not create a PHATE plot after training")
parser.add_argument(
    "--phate_color_by",
    type=str,
    default="label",
    help="Label or population-cache metadata field used to color the PHATE plot",
)
parser.add_argument("--phate_knn", type=int, default=30)
parser.add_argument(
    "--phate_output",
    type=str,
    default=None,
    help="PHATE PNG path (default: <save_dir>/latent_phate.png)",
)
args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = "cuda"
else:
    args.device = "cpu"

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


def make_collate_fn(phenoGS_dists: np.ndarray | None):
    """Return a collate function that pads point clouds and extracts the
    upper-triangular phenoGS sub-matrix for the batch.

    Parameters
    ----------
    phenoGS_dists : (K, K) numpy array or None
        Full pairwise phenoGS distance matrix over all samples.
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

        arange = torch.arange(input_tensor.shape[1])
        mask = arange.unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)

        if phenoGS_dists is not None:
            B = len(indices)
            sub_mat = phenoGS_dists[np.ix_(indices, indices)]  # (B, B)
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = vars(args)
    config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")

    # Labels are retained for PHATE coloring but are not training targets.
    PCs, labels, _ = load_data(
        args.raw_dir,
        args.full,
        population_cache=args.population_cache,
        pdo_caf_cache=args.pdo_caf_cache,
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
    # Pre-compute phenoGS pairwise distances (once, before training)
    # ------------------------------------------------------------------
    phenoGS_dists = None
    if args.dist_weight > 0.0:
        if args.phenoGS_cache and os.path.isfile(args.phenoGS_cache):
            print(f"Loading cached phenoGS distances from {args.phenoGS_cache}")
            phenoGS_dists = np.load(args.phenoGS_cache)
        else:
            raise FileNotFoundError(
                "Precomputed phenoGS distances were not found. "
                "Generate them with the Pheno-GS preparation pipeline and pass --phenoGS_cache."
            )
        if phenoGS_dists.shape != (len(PCs), len(PCs)):
            raise ValueError(
                f"Distance matrix shape {phenoGS_dists.shape} does not match {len(PCs)} populations"
            )
        if not np.isfinite(phenoGS_dists).all() or not np.allclose(phenoGS_dists, phenoGS_dists.T):
            raise ValueError("Distance matrix must be finite and symmetric")

    # ------------------------------------------------------------------
    # Dataset / dataloaders
    # ------------------------------------------------------------------
    max_points = max(pc.shape[0] for pc in PCs)
    dataset = PointCloudDataset(PCs)
    if len(dataset) < 2:
        raise ValueError("At least two populations are required for training")
    test_indices = []
    if args.split_field is not None:
        split_cache_path = args.split_cache or args.population_cache or args.pdo_caf_cache
        if split_cache_path is None:
            raise ValueError("--split_cache is required with --split_field")
        with np.load(split_cache_path, allow_pickle=True) as split_cache:
            group_keys = [str(value) for value in split_cache["group_keys"]]
            matches = [
                i for i, key in enumerate(group_keys)
                if key.lower() == args.split_field.lower()
            ]
            if not matches:
                raise ValueError(
                    f"Split field {args.split_field!r} not found in {group_keys}"
                )
            field_index = matches[0]
            split_values = np.asarray([
                str(name).split("__")[field_index]
                for name in split_cache["group_names"]
            ])
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
            subset_rng.choice(train_indices, args.train_max_samples, replace=False).tolist()
        )
    if args.val_max_samples is not None and len(val_indices) > args.val_max_samples:
        val_indices = sorted(
            subset_rng.choice(val_indices, args.val_max_samples, replace=False).tolist()
        )
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    collate = make_collate_fn(phenoGS_dists)

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
        simplicial_max_neighbors=args.simplicial_max_neighbors,
        simplicial_max_triangles=args.simplicial_max_triangles,
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
    checkpoint_path = os.path.join(args.save_dir, "best_model.pt")
    print(f"Checkpoint saved to: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device, weights_only=True))
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
                "dist_weight": args.dist_weight,
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

    if not args.skip_phate:
        from utils.latent_space import categories_from_cache, plot_phate

        cache_path = args.population_cache or args.pdo_caf_cache
        if cache_path is None and os.path.isfile(args.raw_dir) and args.raw_dir.endswith(".npz"):
            cache_path = args.raw_dir
        categories, category_name = categories_from_cache(
            cache_path,
            args.phate_color_by,
            labels_np,
        )
        phate_path = args.phate_output or os.path.join(args.save_dir, "latent_phate.png")
        embedding = plot_phate(
            all_z,
            categories,
            phate_path,
            category_name=category_name,
            knn=args.phate_knn,
            seed=args.seed,
            n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        )
        np.save(os.path.join(args.save_dir, "phate_embedding.npy"), embedding)
        print(f"PHATE plot saved to: {phate_path}")


if __name__ == "__main__":
    main()
