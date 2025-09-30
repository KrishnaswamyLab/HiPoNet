import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb
from utils.read_data import load_data
from utils.training import collate_fn

from models.graph_learning import HiPoNet, MLP
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
parser.add_argument("--J", type=int, default=3, help="Number of wavelet scales")
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
        for batch, mask, labels in loader:
            logits = model(batch, mask)
            labels = labels.to(logits.device)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels).detach().float().item()
            total += len(labels)
    return (correct * 100) / total


def train(model: nn.Module, PCs, labels):
    print(args)
    opt = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2)
    train_loader = DataLoader(
        [(PCs[i], labels[i]) for i in train_idx],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        [(PCs[i], labels[i]) for i in test_idx],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    total_n_batches = len(train_loader)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0
    with tqdm(range(args.num_epochs)) as tq:
        for epoch in tq:
            correct_train = 0
            t_loss = 0
            model.train()
            opt.zero_grad()
            minibatches_per_batch = args.n_accumulate
            for i, (batch, mask, labels) in enumerate(train_loader, start=1):
                logits = model(batch, mask)
                labels = labels.to(logits.device)
                preds = torch.argmax(logits, dim=1)
                correct_train += torch.sum(preds == labels).detach().float().item()
                loss = loss_fn(logits, labels)
                if args.orthogonal:
                    alphas = model.module[0].layer.alphas
                    loss += (
                        0.1
                        * (
                            alphas @ alphas.T
                            - torch.eye(args.num_weights).to(args.device)
                        )
                        .square()
                        .mean()
                    )
                loss /= minibatches_per_batch
                t_loss += loss.detach().item()
                loss.backward()

                if (i % args.n_accumulate == 0) or i == total_n_batches:
                    opt.step()
                    opt.zero_grad()
                    minibatches_per_batch = min(args.n_accumulate, total_n_batches - i)

                del (logits, loss, preds)
                torch.cuda.empty_cache()
                gc.collect()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch + 1)

            train_acc = correct_train * 100 / len(train_idx)
            test_acc = test(model, test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
            wandb.log(
                {
                    "Loss": t_loss,
                    "Train acc": train_acc,
                    "Test acc": test_acc,
                    "Best acc": best_acc,
                },
                step=epoch + 1,
            )

            tq.set_description(
                "Train Loss = %.4f, Train acc = %.4f, Test acc = %.4f, Best acc = %.4f"
                % (t_loss, train_acc, test_acc, best_acc)
            )
    print(f"Best accuracy : {best_acc}")

class ClassificationModel(nn.Module):
        def __init__(self, hiponet, mlp_classifier):
            super().__init__()
            self.hiponet = hiponet
            self.mlp_classifier = mlp_classifier

        def forward(self, batch, mask):
            out1 = self.hiponet(batch, mask)
            out2 = self.mlp_classifier(out1)
            return out2

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
        args.J,
        args.device,
        args.sigma,
    )
    with torch.no_grad():
        batch = PCs[0].to(args.device)[None, ...]
        mask = batch.sum(-1) != 0
        input_dim = hiponet(PCs[0].to(args.device)[None, ...], mask).shape[1]
    mlp_classifier = MLP(input_dim, args.hidden_dim, num_labels, args.num_layers).to(
        args.device
    )
    model = nn.DataParallel(ClassificationModel(hiponet, mlp_classifier))
    train(model, PCs, labels)


if __name__ == "__main__":
    main()
