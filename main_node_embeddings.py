import torch
from tqdm import tqdm
import wandb
import pathlib
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from models.graph_learning import HiPoNet, MLPAutoEncoder
from argparse import ArgumentParser
from utils.read_data import load_data
from utils.training import save_model

import gc
import os

SMOKE_TEST = os.environ.get("SMOKE_TEST")
WEIGHTS_SAVE_LOC = pathlib.Path(__file__).parent / "model_weights"
if not WEIGHTS_SAVE_LOC.exists():
    WEIGHTS_SAVE_LOC.mkdir()

gc.enable()

# Define the parameters using parser args
parser = ArgumentParser(description="Pointcloud net")
parser.add_argument(
    "--raw_dir",
    type=str,
    default="data/sea",
    help="Directory where the raw data is stored",
)
parser.add_argument("--full", action="store_true")
parser.add_argument("--task", type=str, default="prolif", help="Task on PDO data")
parser.add_argument(
    "--spatial_threshold",
    type=float,
    default=0.5,
    help="Threshold for creating the graph",
)
parser.add_argument(
    "--gene_threshold", type=float, default=0.5, help="Threshold for creating the graph"
)
parser.add_argument("--sigma", type=float, default=0.5, help="Bandwidth")
parser.add_argument("--K", type=int, default=1, help="Order of simplicial complex")
parser.add_argument("--J", type=int, default=3, help="Order of simplicial complex")
parser.add_argument(
    "--hidden_dim", type=int, default=256, help="Hidden dim for the MLP Autoencoder"
)
parser.add_argument(
    "--embedding_dim", type=int, default=4, help="Autoencoder embedding dimension"
)
parser.add_argument("--num_layers", type=int, default=3, help="Number of MLP layers")
parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
parser.add_argument("--wd", type=float, default=3e-3, help="Weight decay")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--gpu", type=int, default=0, help="GPU index")
parser.add_argument("--disable_wb", action="store_true", help="Disable wandb logging")
parser.add_argument(
    "--n_accumulate",
    default=1,
    type=int,
    help="number of batches to accumulate gradients over",
)
args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    print(f"Using {torch.cuda.device_count()} GPUs")
    args.device = "cuda"
else:
    args.device = "cpu"


def collate_fn(batch):
    gene = torch.nested.as_nested_tensor(
        [x[0] for x in batch], layout=torch.jagged
    ).to_padded_tensor(padding=0.0)
    gene_mask = gene.sum(-1) != 0
    spatial = torch.nested.as_nested_tensor(
        [x[1] for x in batch], layout=torch.jagged
    ).to_padded_tensor(padding=0.0)
    spatial_mask = spatial.sum(-1) != 0
    return gene, gene_mask, spatial, spatial_mask


def test(
    model_gene: HiPoNet,
    model_spatial: HiPoNet,
    autoenc: MLPAutoEncoder,
    test_loader: DataLoader,
):
    model_gene.eval(), model_spatial.eval(), autoenc.eval()
    total_loss = 0
    weight_sum = 0
    with torch.no_grad():
        for batch_gene, mask_gene, batch_spatial, mask_spatial in test_loader:
            batch_gene, mask_gene, batch_spatial, mask_spatial = (
                batch_gene.to(args.device),
                mask_gene.to(args.device),
                batch_spatial.to(args.device),
                mask_spatial.to(args.device),
            )
            X_spatial, X_gene = (
                model_spatial(batch_spatial, mask_spatial),
                model_gene(batch_gene, mask_gene),
            )
            # Embedding of shape (n_nodes, n_spatial_embedding_dims + n_gene_embedding_dims)
            embedding = torch.cat([X_spatial, X_gene], 1)
            reconstructed = autoenc(embedding)
            points_per_cloud = (mask_gene * mask_gene.sum(1, keepdim=True))[mask_gene]
            weights = points_per_cloud
            loss = (
                weights
                * torch.nn.functional.mse_loss(
                    reconstructed, embedding, reduction="none"
                ).sum(1)  # Sum over feature dim
            ).sum()
            total_loss += loss.detach()
            weight_sum += weights.sum()
            torch.cuda.empty_cache()
            gc.collect()

    return total_loss / weight_sum


def train(
    model_gene: HiPoNet,
    model_spatial: HiPoNet,
    autoenc: MLPAutoEncoder,
    PC_gene: torch.tensor,
    PC_spatial: torch.tensor,
    weights_save_loc: pathlib.Path | None = None,
):
    print(args)
    opt = torch.optim.AdamW(
        list(model_gene.parameters())
        + list(model_spatial.parameters())
        + list(autoenc.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    train_idx, test_idx = train_test_split(np.arange(len(PC_gene)), test_size=0.2)
    train_loader = DataLoader(
        [(PC_gene[i], PC_spatial[i]) for i in train_idx],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        [(PC_gene[i], PC_spatial[i]) for i in test_idx],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    total_n_batches = len(train_loader)
    best_test_loss = float("inf")
    with tqdm(range(args.num_epochs)) as tq:
        for epoch in tq:
            model_gene.train()
            model_spatial.train()
            autoenc.train()
            opt.zero_grad()
            minibatches_per_batch = args.n_accumulate
            for i, (batch_gene, mask_gene, batch_spatial, mask_spatial) in enumerate(
                train_loader, start=1
            ):
                batch_gene, mask_gene, batch_spatial, mask_spatial = (
                    batch_gene.to(args.device),
                    mask_gene.to(args.device),
                    batch_spatial.to(args.device),
                    mask_spatial.to(args.device),
                )
                X_spatial, X_gene = (
                    model_spatial(batch_spatial, mask_spatial),
                    model_gene(batch_gene, mask_gene),
                )
                # Embedding of shape (n_nodes, n_spatial_embedding_dims + n_gene_embedding_dims)
                embedding = torch.cat([X_spatial, X_gene], 1)
                reconstructed = autoenc(embedding)

                # We don't want to naively average over all nodes - we want to do weighted average based on
                # This ensures we weight each *point cloud* equally (instead of each node)
                points_per_cloud = (mask_gene * mask_gene.sum(1, keepdim=True))[
                    mask_gene
                ]
                # Weights sum to 1
                weights = points_per_cloud / points_per_cloud.sum()
                loss = (
                    weights
                    * torch.nn.functional.mse_loss(
                        reconstructed, embedding, reduction="none"
                    ).sum(1)  # Sum over feature dim
                ).sum()

                loss /= minibatches_per_batch
                loss.backward()

                if (i % args.n_accumulate == 0) or i == total_n_batches:
                    for model in [model_spatial, model_gene, autoenc]:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                wandb.log(
                                    {f"{name}.grad": param.grad.norm()}, step=epoch + 1
                                )
                    opt.step()
                    opt.zero_grad()
                    minibatches_per_batch = min(args.n_accumulate, total_n_batches - i)

                del (embedding, reconstructed)
                torch.cuda.empty_cache()
                gc.collect()

            test_loss = test(model_gene, model_spatial, autoenc, test_loader)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                for model, name in [
                    (model_gene, "model_gene"),
                    (model_spatial, "model_spatial"),
                    (autoenc, "autoenc"),
                ]:
                    save_model(model, name, weights_save_loc)
                torch.save(
                    {
                        "train_idx": torch.tensor(train_idx),
                        "test_idx": torch.tensor(test_idx),
                    },
                    weights_save_loc / "split_idx.pt",
                )

            torch.cuda.empty_cache()
            gc.collect()

            loss_float = loss.detach().item()
            wandb.log(
                {
                    "train_loss": loss_float,
                    "test_loss": test_loss,
                    "best_test_loss": best_test_loss,
                },
                step=epoch + 1,
            )

            tq.set_description(
                f"Train Loss = {loss_float:.4f}, Test Loss = {test_loss:.4f}"
            )


def main():
    import os

    config = vars(args)
    config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")
    wandb.init(
        project="pointcloud-net-node-embeddings",
        config=config,
        mode="disabled" if args.disable_wb else None,
    )

    PC_gene, PC_spatial = load_data(args.raw_dir, args.full)
    model_spatial = (
        HiPoNet(
            dimension=PC_spatial[0].shape[1],
            n_weights=1,
            threshold=args.spatial_threshold,
            K=args.K,
            J=args.J,
            device=args.device,
            sigma=args.sigma,
            pooling=False,
        )
        .to(args.device)
        .float()
    )
    model_gene = (
        HiPoNet(
            dimension=PC_gene[0].shape[1],
            n_weights=1,
            threshold=args.gene_threshold,
            K=args.K,
            J=args.J,
            device=args.device,
            sigma=args.sigma,
            pooling=False,
        )
        .to(args.device)
        .float()
    )
    with torch.no_grad():
        input_dim = (
            model_spatial(
                PC_spatial[0][:5].unsqueeze(0).to(args.device),
                torch.zeros((1, 5), dtype=torch.bool).to(args.device),
            ).shape[1]
            + model_gene(
                PC_gene[0][:5].unsqueeze(0).to(args.device),
                torch.zeros((1, 5), dtype=torch.bool).to(args.device),
            ).shape[1]
        )
    if SMOKE_TEST:
        PC_gene, PC_spatial = (
            [PC_gene[i][: 100 + i] for i in range(20)],
            [PC_spatial[i][: 100 + i] for i in range(20)],
        )
        weights_save_loc = None
    autoencoder = MLPAutoEncoder(
        input_dim, args.hidden_dim, args.embedding_dim, args.num_layers, bn=False
    ).to(args.device)

    weights_save_loc = WEIGHTS_SAVE_LOC / config["slurm_job_id"]
    weights_save_loc.mkdir(exist_ok=True)
    train(model_gene, model_spatial, autoencoder, PC_gene, PC_spatial, weights_save_loc)


if __name__ == "__main__":
    main()
