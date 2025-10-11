import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import pathlib
from utils.read_data import load_data
from utils.training import collate_fn, save_model

from models.graph_learning import HiPoNet, MLPAutoEncoder
from argparse import ArgumentParser

PRECOMPUTED_EMBEDDINGS_LOC = (
    pathlib.Path(__file__).parent / "data" / "precomputed_embeddings"
)
WEIGHTS_SAVE_LOC = pathlib.Path(__file__).parent / "model_weights"
if not WEIGHTS_SAVE_LOC.exists():
    WEIGHTS_SAVE_LOC.mkdir()

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
parser.add_argument(
    "--threshold", type=float, default=0.5, help="Threshold for creating the graph"
)
parser.add_argument("--sigma", type=float, default=0.5, help="Bandwidth")
parser.add_argument("--K", type=int, default=1, help="Order of simplicial complex")
parser.add_argument("--J", type=int, default=3, help="Number of wavelet scales")
parser.add_argument(
    "--hidden_dim", type=int, default=512, help="Hidden dim for the MLP"
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
parser.add_argument(
    "--orthogonal",
    action="store_true",
    help="If set, use orthogonality loss on the alpha parameter",
)
parser.add_argument(
    "--embedding_dim", type=int, default=128, help="Autoencoder embedding dimension"
)
parser.add_argument("--regenerate_embeddings", action="store_true")
args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    print(f"Using {torch.cuda.device_count()} GPUs")
    args.device = "cuda"
else:
    args.device = "cpu"


def prepare_dataset(hiponet: HiPoNet, PCs, raw_dir: str):
    """Precompute hiponet embeddings."""
    save_loc = (
        PRECOMPUTED_EMBEDDINGS_LOC / f"{raw_dir.rstrip('/').split('/')[-1]}_emb.pt"
    )
    if save_loc.exists() and not args.recompute_embeddings:
        embeddings = torch.load(save_loc, map_location="cpu")
    else:
        full_loader = DataLoader(
            list(zip(PCs, range(len(PCs)))),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )
        hiponet.eval()
        all_embeddings = []
        with torch.no_grad():
            for batch, mask, _ in full_loader:
                batch, mask = batch.to(args.device), mask.to(args.device)
                hn_embeddings = hiponet(batch, mask).to("cpu")
                all_embeddings.append(hn_embeddings)
        embeddings = torch.concat(all_embeddings, 0)
        # Normalize across dimensions so that we don't concentrate only on one
        embeddings -= embeddings.mean(dim=0, keepdim=True)
        embeddings /= embeddings.std(dim=0, keepdim=True)
        torch.save(embeddings, save_loc)

    embeddings_dataset = torch.utils.data.TensorDataset(embeddings)
    train_data, test_data = torch.utils.data.random_split(
        embeddings_dataset, lengths=[0.8, 0.2]
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print("Precomputed Embeddings")
    return train_loader, test_loader


def test(model, loader):
    model.eval()
    test_loss = 0
    count = 0
    with torch.no_grad():
        for (hn_embedding,) in loader:
            hn_embedding = hn_embedding.to(args.device)
            reconstructed = model(hn_embedding)
            test_loss += torch.nn.functional.mse_loss(
                reconstructed, hn_embedding, reduction="sum"
            )
            count += hn_embedding.shape[0]
    return test_loss / count


def train(hiponet, mlp_autoencoder: nn.Module, PCs, weights_save_loc, raw_dir):
    train_loader, test_loader = prepare_dataset(hiponet, PCs, raw_dir)
    opt = torch.optim.AdamW(
        list(mlp_autoencoder.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )

    total_n_batches = len(train_loader)
    loss_fn = torch.nn.MSELoss()
    best_loss = float("inf")
    with tqdm(range(args.num_epochs)) as tq:
        for epoch in tq:
            train_loss = 0
            mlp_autoencoder.train()
            opt.zero_grad()
            minibatches_per_batch = args.n_accumulate
            for i, (hn_embedding,) in enumerate(train_loader, start=1):
                hn_embedding = hn_embedding.to(args.device)
                reconstructed = mlp_autoencoder(hn_embedding)
                loss = loss_fn(reconstructed, hn_embedding)
                loss /= minibatches_per_batch
                train_loss += loss.detach().item()
                loss.backward()

                if (i % args.n_accumulate == 0) or i == total_n_batches:
                    opt.step()
                    opt.zero_grad()
                    minibatches_per_batch = min(args.n_accumulate, total_n_batches - i)

            for name, param in mlp_autoencoder.named_parameters():
                if param.grad is not None:
                    wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch + 1)

            test_loss = test(mlp_autoencoder, test_loader)
            if test_loss < best_loss:
                best_loss = test_loss
                save_model(mlp_autoencoder, "autoencoder", weights_save_loc)
            wandb.log(
                {
                    "train loss": train_loss,
                    "test loss": test_loss,
                    "best loss": best_loss,
                },
                step=epoch + 1,
            )

            tq.set_description(
                "Train Loss = %.4f, Test Loss = %.4f, Best Loss = %.4f"
                % (train_loss, test_loss, best_loss)
            )
    print(f"Best loss : {best_loss}")


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
        1,
        args.threshold,
        args.K,
        args.J,
        args.device,
        args.sigma,
        ignore_alphas=True,
    )
    with torch.no_grad():
        batch = PCs[0].to(args.device)[None, ...]
        mask = batch.sum(-1) != 0
        input_dim = hiponet(PCs[0].to(args.device)[None, ...], mask).shape[1]
    mlp_autoencoder = MLPAutoEncoder(
        input_dim, args.hidden_dim, args.embedding_dim, args.num_layers
    ).to(args.device)
    weights_save_loc = WEIGHTS_SAVE_LOC / config["slurm_job_id"]
    weights_save_loc.mkdir(exist_ok=True)
    train(hiponet, mlp_autoencoder, PCs, weights_save_loc, args.raw_dir)


if __name__ == "__main__":
    main()
