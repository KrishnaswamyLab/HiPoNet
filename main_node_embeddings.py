import torch
from tqdm import tqdm
import wandb
from utils.read_data import load_data

from models.graph_learning import HiPoNet, MLPAutoEncoder
from argparse import ArgumentParser

import gc

gc.enable()

# Define the parameters using parser args
parser = ArgumentParser(description="Pointcloud net")
parser.add_argument(
    "--raw_dir",
    type=str,
    default="COVID_data",
    help="Directory where the raw data is stored",
)
parser.add_argument("--full", action="store_true")
parser.add_argument("--task", type=str, default="prolif", help="Task on PDO data")
parser.add_argument("--num_weights", type=int, default=2, help="Number of weights")
parser.add_argument(
    "--threshold", type=float, default=0.5, help="Threshold for creating the graph"
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


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, labels in loader:
            logits = model(batch)
            labels = labels.to(logits.device)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels).detach().float().item()
            total += len(labels)
    return (correct * 100) / total


def train(hiponet: HiPoNet, autoenc: MLPAutoEncoder, PC, node_labels):
    print(args)
    opt = torch.optim.AdamW(
        list(hiponet.parameters()) + list(autoenc.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    loss_fn = torch.nn.MSELoss()
    with tqdm(range(args.num_epochs)) as tq:
        for epoch in tq:
            hiponet.train()
            autoenc.train()
            opt.zero_grad()

            hiponet_embedding = hiponet(PC)
            reconstructed = autoenc(hiponet)
            loss = loss_fn(hiponet_embedding, reconstructed)
            loss.backward()

            torch.cuda.empty_cache()
            gc.collect()

            for name, param in hiponet.named_parameters():
                if param.grad is not None:
                    wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch + 1)
            for name, param in autoenc.named_parameters():
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

    assert args.batch_size % 2 == 0, "Batch size must be even"
    args.effective_batch_size = args.batch_size * args.n_accumulate

    config = vars(args)
    config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")
    wandb.init(
        project="pointcloud-net-k-fold",
        config=config,
        mode="disabled" if args.disable_wb else None,
    )

    PCs, labels, num_labels = load_data(args.raw_dir, args.full)
    hiponet = HiPoNet(
        PCs[0].shape[1],
        args.num_weights,
        args.threshold,
        args.K,
        args.device,
        args.sigma,
    )
    with torch.no_grad():
        input_dim = hiponet(PCs[0].to(args.device)[None, ...]).shape[1]
    autoencoder = MLPAutoEncoder(
        input_dim, args.hidden_dim, num_labels, args.num_layers
    ).to(args.device)
    train(hiponet, autoencoder, PCs, labels)


if __name__ == "__main__":
    main()
