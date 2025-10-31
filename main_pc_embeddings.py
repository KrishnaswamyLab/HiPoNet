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
parser.add_argument("--learn_alphas", action="store_true")
parser.add_argument("--alpha_loss_weight", type=float, default=0.1)
args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    print(f"Using {torch.cuda.device_count()} GPUs")
    args.device = "cuda"
else:
    args.device = "cpu"


def prepare_dataset(hiponet: HiPoNet, PCs, labels, raw_dir: str):
    """Precompute hiponet embeddings."""
    emb_save_loc = (
        PRECOMPUTED_EMBEDDINGS_LOC / f"{raw_dir.rstrip('/').split('/')[-1]}_emb.pt"
    )
    label_save_loc = (
        PRECOMPUTED_EMBEDDINGS_LOC / f"{raw_dir.rstrip('/').split('/')[-1]}_label.pt"
    )
    batch_save_loc = (
        PRECOMPUTED_EMBEDDINGS_LOC / f"{raw_dir.rstrip('/').split('/')[-1]}_batch.pt"
    )
    mask_save_loc = (
        PRECOMPUTED_EMBEDDINGS_LOC / f"{raw_dir.rstrip('/').split('/')[-1]}_mask.pt"
    )
    if emb_save_loc.exists() and not args.regenerate_embeddings:
        embeddings = torch.load(emb_save_loc, map_location="cpu")
        batches = torch.load(batch_save_loc, map_location="cpu")
        masks = torch.load(mask_save_loc, map_location="cpu")
    else:
        full_loader = DataLoader(
            list(zip(PCs, labels)),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )
        hiponet.eval()
        # The pre-generated embeddings are *without alphas*
        ignore_alphas = hiponet.layer.ignore_alphas
        hiponet.layer.ignore_alphas = True
        all_embeddings = []
        all_labels = []
        all_batches = []
        all_masks = []
        with torch.no_grad():
            for batch, mask, labels in full_loader:
                all_batches.append(batch)
                all_masks.append(mask)
                batch, mask = batch.to(args.device), mask.to(args.device)
                hn_embeddings = hiponet(batch, mask).to("cpu")
                all_embeddings.append(hn_embeddings)
                all_labels.append(labels)

        embeddings = torch.concat(all_embeddings, 0)
        # Normalize across dimensions so that we don't concentrate only on one
        embeddings -= embeddings.mean(dim=0, keepdim=True)
        embeddings /= embeddings.std(dim=0, keepdim=True)
        torch.save(embeddings, emb_save_loc)

        batches, masks = (
            torch.concat(all_batches, dim=0),
            torch.concat(all_masks, dim=0),
        )
        torch.save(torch.concat(all_labels, dim=0), label_save_loc)
        torch.save(batches, batch_save_loc)
        torch.save(masks, mask_save_loc)

        # Reset to previous
        hiponet.layer.ignore_alphas = ignore_alphas

    embeddings_dataset = torch.utils.data.TensorDataset(embeddings, batches, masks)
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
        for hn_embedding, batch, mask in loader:
            hn_embedding = hn_embedding.to(args.device)
            reconstructed = model(hn_embedding)
            test_loss += (
                torch.nn.functional.mse_loss(
                    reconstructed, hn_embedding, reduction="mean"
                )
                * hn_embedding.shape[0]
            )
            count += hn_embedding.shape[0]
    return test_loss / count


def train(
    hiponet: HiPoNet, mlp_autoencoder: nn.Module, PCs, labels, weights_save_loc, raw_dir
):
    train_loader, test_loader = prepare_dataset(hiponet, PCs, labels, raw_dir)
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
            count = 0
            mlp_autoencoder.train()
            opt.zero_grad()
            minibatches_per_batch = args.n_accumulate
            for i, (hn_embedding, batch, mask) in enumerate(train_loader, start=1):
                hn_embedding = hn_embedding.to(args.device)

                if args.learn_alphas:
                    batch = batch.to(args.device)
                    mask = mask.to(args.device)
                    input_embedding = hiponet(batch, mask)
                else:
                    input_embedding = hn_embedding

                reconstructed = mlp_autoencoder(input_embedding)
                loss = loss_fn(reconstructed, hn_embedding)

                alpha_loss = torch.nan
                if args.learn_alphas:
                    alpha_loss = -1 * (
                        args.alpha_loss_weight
                        * torch.softmax(hiponet.layer.alphas, dim=1).pow(2).sum()
                    )
                    loss += alpha_loss

                loss /= minibatches_per_batch
                train_loss += loss.detach().item() * hn_embedding.shape[0]
                count += hn_embedding.shape[0]
                loss.backward()

                if (i % args.n_accumulate == 0) or i == total_n_batches:
                    opt.step()
                    opt.zero_grad()
                    minibatches_per_batch = min(args.n_accumulate, total_n_batches - i)

            for name, param in mlp_autoencoder.named_parameters():
                if param.grad is not None:
                    wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch + 1)

            train_loss /= count

            test_loss = test(mlp_autoencoder, test_loader)
            if test_loss < best_loss:
                best_loss = test_loss
                save_model(mlp_autoencoder, "autoencoder", weights_save_loc)
            wandb.log(
                {
                    "train loss": train_loss,
                    "test loss": test_loss,
                    "best loss": best_loss,
                    "alpha_loss": alpha_loss,
                },
                step=epoch + 1,
            )

            desc = "Train Loss = %.4f, Test Loss = %.4f, Best Loss = %.4f" % (
                train_loss,
                test_loss,
                best_loss,
            )
            if args.learn_alphas:
                desc += f", Alpha loss = {alpha_loss:.4f}"
            tq.set_description(desc)
    print(f"Best loss : {best_loss}")


def main():
    import os

    assert args.batch_size % 2 == 0, "Batch size must be even"
    args.effective_batch_size = args.batch_size * args.n_accumulate

    config = vars(args)
    config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")

    PCs, labels, num_labels = load_data(args.raw_dir, args.full)
    if os.environ.get("SMOKE_TEST"):
        PCs = [pc[:100] for pc in PCs[:100]]
        labels = labels[:100]
        args.disable_wb = True

    wandb.init(
        project="pointcloud-net-k-fold",
        config=config,
        mode="disabled" if args.disable_wb else None,
    )

    hiponet = HiPoNet(
        PCs[0].shape[1],
        1,
        args.threshold,
        args.K,
        args.J,
        args.device,
        args.sigma,
        ignore_alphas=(not args.learn_alphas),
        softmax_alphas=args.learn_alphas,
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
    train(hiponet, mlp_autoencoder, PCs, labels, weights_save_loc, args.raw_dir)


if __name__ == "__main__":
    main()
