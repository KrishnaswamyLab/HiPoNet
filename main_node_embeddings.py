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
    "--reconstruct_original",
    action="store_true",
    help="If true, reconstruction loss is wrt original point cloud, not the wavelet coefficients",
)
parser.add_argument(
    "--ignore_alphas", action="store_true", help="Fix the alpha params to 1"
)
parser.add_argument(
    "--alpha_connectivity_only",
    action="store_true",
    help="only use alphas for connectivity, not node features",
)
parser.add_argument(
    "--normalize_alphas",
    action="store_true",
)
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


class NodeEmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        gene_model: HiPoNet,
        spatial_model: HiPoNet,
        autoencoder: MLPAutoEncoder,
        num_embedding_features: int,
    ):
        super().__init__()
        self.gene_model = gene_model
        self.spatial_model = spatial_model
        # We need to add batchnorm, otherwise the model can just learn to make things really close to 0
        self.bn = torch.nn.BatchNorm1d(num_embedding_features)
        self.autoencoder = autoencoder

    def forward(self, gene_batch, gene_mask, spatial_batch, spatial_mask):
        gene_embedding = self.gene_model(gene_batch, gene_mask)
        spatial_embedding = self.spatial_model(spatial_batch, spatial_mask)
        embedding = self.bn(torch.cat([gene_embedding, spatial_embedding], 1))
        reconstructed_embedding = self.autoencoder(embedding)
        return embedding, reconstructed_embedding

    def encode(self, gene_batch, gene_mask, spatial_batch, spatial_mask):
        gene_embedding = self.gene_model(gene_batch, gene_mask)
        spatial_embedding = self.spatial_model(spatial_batch, spatial_mask)
        embedding = self.bn(torch.cat([gene_embedding, spatial_embedding], 1))
        return self.autoencoder.encode(embedding)


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
    model: NodeEmbeddingModel,
    test_loader: DataLoader,
    reconstruct_original: bool,
):
    model.eval()
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
            embedding, reconstructed = model(
                batch_gene, mask_gene, batch_spatial, mask_spatial
            )
            points_per_cloud = (mask_gene * mask_gene.sum(1, keepdim=True))[mask_gene]
            weights = points_per_cloud
            target = (
                torch.cat((batch_gene[mask_gene], batch_spatial[mask_spatial]), dim=1)
                if reconstruct_original
                else embedding
            )
            loss = (
                weights
                * torch.nn.functional.mse_loss(
                    reconstructed, target, reduction="none"
                ).sum(1)  # Sum over feature dim
            ).sum()
            total_loss += loss.detach()
            weight_sum += weights.sum()
            torch.cuda.empty_cache()
            gc.collect()

    return total_loss / weight_sum


def train(
    model: NodeEmbeddingModel,
    PC_gene: torch.tensor,
    PC_spatial: torch.tensor,
    reconstruct_original: bool,
    weights_save_loc: pathlib.Path | None = None,
):
    print(args)
    opt = torch.optim.AdamW(
        list(model.parameters()),
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
            model.train()
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
                embedding, reconstructed = model(
                    batch_gene, mask_gene, batch_spatial, mask_spatial
                )

                # We don't want to naively average over all nodes - we want to do weighted average based on
                # This ensures we weight each *point cloud* equally (instead of each node)
                points_per_cloud = (mask_gene * mask_gene.sum(1, keepdim=True))[
                    mask_gene
                ]
                # Weights sum to 1
                weights = points_per_cloud / points_per_cloud.sum()
                target = (
                    torch.cat(
                        (batch_gene[mask_gene], batch_spatial[mask_spatial]), dim=1
                    )
                    if reconstruct_original
                    else embedding
                )
                loss = (
                    weights
                    * torch.nn.functional.mse_loss(
                        reconstructed, target, reduction="none"
                    ).sum(1)  # Sum over feature dim
                ).sum()

                loss /= minibatches_per_batch
                loss.backward()

                if (i % args.n_accumulate == 0) or i == total_n_batches:
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

            test_loss = test(model, test_loader, reconstruct_original)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_model(model, "model", weights_save_loc)
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

    PC_gene, PC_spatial, _ = load_data(args.raw_dir, args.full)
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
            normalize_alphas=args.normalize_alphas,
            use_alphas_for_connectivity_only=args.alpha_connectivity_only,
            ignore_alphas=args.ignore_alphas,
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
            normalize_alphas=args.normalize_alphas,
            use_alphas_for_connectivity_only=args.alpha_connectivity_only,
            ignore_alphas=args.ignore_alphas,
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

    output_dim = (
        PC_gene[0].shape[1] + PC_spatial[0].shape[1]
        if args.reconstruct_original
        else input_dim
    )
    autoencoder = MLPAutoEncoder(
        input_dim,
        args.hidden_dim,
        args.embedding_dim,
        args.num_layers,
        bn=False,
        output_dim=output_dim,
    ).to(args.device)

    model = NodeEmbeddingModel(model_gene, model_spatial, autoencoder, input_dim).to(
        args.device
    )

    weights_save_loc = WEIGHTS_SAVE_LOC / config["slurm_job_id"]
    weights_save_loc.mkdir(exist_ok=True)
    train(model, PC_gene, PC_spatial, args.reconstruct_original, weights_save_loc)


if __name__ == "__main__":
    main()
