import torch
from tqdm import tqdm
import wandb
from utils.read_data import load_data

from models.graph_learning import HiPoNet, MLPAutoEncoder
from argparse import ArgumentParser

import gc
import os
SMOKE_TEST = os.environ.get("SMOKE_TEST")

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
parser.add_argument("--num_weights", type=int, default=2, help="Number of weights")
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
parser.add_argument(
    "--hidden_dim", type=int, default=256, help="Hidden dim for the MLP"
)
parser.add_argument("--num_layers", type=int, default=3, help="Number of MLP layers")
parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
parser.add_argument("--wd", type=float, default=3e-3, help="Weight decay")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--gpu", type=int, default=0, help="GPU index")
parser.add_argument("--disable_wb", action="store_true", help="Disable wandb logging")
args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    print(f"Using {torch.cuda.device_count()} GPUs")
    args.device = "cuda"
else:
    args.device = "cpu"


def train(
    model_gene: HiPoNet,
    model_spatial: HiPoNet,
    autoenc: MLPAutoEncoder,
    PC_gene: torch.tensor,
    PC_spatial: torch.tensor,
):
    print(args)
    opt = torch.optim.AdamW(
        list(model_gene.parameters())
        + list(model_spatial.parameters())
        + list(autoenc.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    loss_fn = torch.nn.MSELoss()
    with tqdm(range(args.num_epochs)) as tq:
        for epoch in tq:
            model_gene.train()
            model_spatial.train()
            autoenc.train()
            opt.zero_grad()

            X_spatial, X_gene = model_spatial([PC_spatial]), model_gene([PC_gene])
            embedding = torch.cat([X_spatial, X_gene], 1)
            reconstructed = autoenc(embedding)
            loss = loss_fn(embedding, reconstructed)
            loss.backward()

            torch.cuda.empty_cache()
            gc.collect()

            for model in [model_spatial, model_gene, autoenc]:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch + 1)

            loss_float = loss.detach().item()
            wandb.log(
                {
                    "Loss": loss_float,
                },
                step=epoch + 1,
            )

            tq.set_description(f"Train Loss = {loss_float:.4f}")


def main():
    import os

    config = vars(args)
    config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")
    wandb.init(
        project="pointcloud-net-k-fold",
        config=config,
        mode="disabled" if args.disable_wb else None,
    )

    PC_spatial, PC_gene, labels, num_labels = load_data(args.raw_dir, args.full)
    model_spatial = (
        HiPoNet(
            dimension=PC_spatial[0].shape[1],
            num_weights=1,
            threshold=args.gene_threshold,
            K=args.K,
            device=args.device,
            sigma=args.sigma,
        )
        .to(args.device)
        .float()
    )
    model_gene = (
        HiPoNet(
            dimension=PC_gene[0].shape[1],
            num_weights=1,
            threshold=args.gene_threshold,
            K=args.K,
            device=args.device,
            sigma=args.sigma,
        )
        .to(args.device)
        .float()
    )
    with torch.no_grad():
        input_dim = (
            model_spatial([PC_spatial[:5].to(args.device)]).shape[1]
            + model_gene([PC_gene[:5].to(args.device)]).shape[1]
        )
    if SMOKE_TEST:
        PC_gene, PC_spatial = PC_gene[:100], PC_spatial[:100]
    autoencoder = MLPAutoEncoder(
        input_dim, args.hidden_dim, num_labels, args.num_layers
    ).to(args.device)
    train(model_gene, model_spatial, autoencoder, PC_gene, PC_spatial)


if __name__ == "__main__":
    main()
