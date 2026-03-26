import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb

from utils.read_data import load_data
from utils.training import collate_fn
from functools import partial

from models.graph_learning import HiPoNet, MLP
from argparse import ArgumentParser

import gc

gc.enable()

# Define the parameters using parser args
parser = ArgumentParser(description="Pointcloud net")
parser.add_argument(
    "--raw_dir",
    type=str,
    default="pdo_data",
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
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
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
    "--sparse",
    action="store_true",
    help="If set, add L1 sparsity loss on alphas to encourage each view to focus on fewer features",
)
parser.add_argument(
    "--sparse_lambda",
    type=float,
    default=0.01,
    help="Weight for the L1 sparsity loss on alphas",
)
parser.add_argument("--transpose", action="store_true")
parser.add_argument(
    "--use_geometric_laplacian",
    action="store_true",
    help="Use metric-aware geometric Hodge Laplacian (requires K >= 2)",
)
parser.add_argument(
    "--diffusion_steps",
    type=int,
    default=1,
    help="Number of diffusion steps t for computing P^t (used for diffusion distances)",
)
parser.add_argument(
    "--use_attention",
    action="store_true",
    help="Use DeepSet attention pooling over simplices (K >= 2). Enables interpretable attention weights.",
)
args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    print(f"Using {torch.cuda.device_count()} GPUs")
    args.device = "cuda"
else:
    args.device = "cpu"


def test(model, mlp, loader):
    model.eval()
    mlp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, mask, labels in loader:
            batch, mask = batch.to(args.device), mask.to(args.device)
            X = model(batch, mask)
            logits = mlp(X)
            labels = labels.to(logits.device)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels).detach().float().item()
            total += len(labels)
    return (correct * 100) / total


def train(model, mlp, PCs, labels):
    print(args)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(mlp.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    # print("Preparing data loaders...")
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2)
    collator = partial(collate_fn, transpose=args.transpose)
    train_loader = DataLoader(
        [(PCs[i], labels[i]) for i in train_idx],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        [(PCs[i], labels[i]) for i in test_idx],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    total_n_batches = len(train_loader)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0
    
    # Log initial alpha values
    for k in range(len(model.layer.alphas)):
        for d in range(len(model.layer.alphas[k])):
            wandb.log({f'Alpha{k}_{d}': model.layer.alphas[k][d].item()}, step=0)
    
    with tqdm(range(args.num_epochs)) as tq:
        for epoch in tq:
            correct_train = 0
            t_loss = 0
            model.train()
            mlp.train()
            opt.zero_grad()
            minibatches_per_batch = args.n_accumulate
            for i, (batch, mask, labels) in enumerate(train_loader, start=1):
                batch, mask = batch.to(args.device), mask.to(args.device)
                X = model(batch, mask)
                logits = mlp(X)
                labels = labels.to(logits.device)
                preds = torch.argmax(logits, dim=1)
                correct_train += torch.sum(preds == labels).detach().float().item()
                loss = loss_fn(logits, labels)
                if args.orthogonal:
                    alphas = model.layer.alphas
                    loss += (
                        0.1
                        * (
                            alphas @ alphas.T
                            - torch.eye(args.num_weights).to(args.device)
                        )
                        .square()
                        .mean()
                    )
                if args.sparse:
                    loss += args.sparse_lambda * model.layer.alphas.abs().sum()
                loss /= minibatches_per_batch
                t_loss += loss.detach().item()
                loss.backward()

                if (i % args.n_accumulate == 0) or i == total_n_batches:
                    opt.step()
                    opt.zero_grad()
                    minibatches_per_batch = min(args.n_accumulate, total_n_batches - i)

                del (X, logits, loss, preds)
                torch.cuda.empty_cache()
                gc.collect()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch + 1)

            train_acc = correct_train * 100 / len(train_idx)
            test_acc = test(model, mlp, test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
            
            # Log alpha values
            for k in range(len(model.layer.alphas)):
                for d in range(len(model.layer.alphas[k])):
                    wandb.log({f'Alpha{k}_{d}': model.layer.alphas[k][d].item()}, step=epoch + 1)
            
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


def main():
    import os

    assert args.batch_size % 2 == 0, "Batch size must be even"
    args.effective_batch_size = args.batch_size * args.n_accumulate

    if args.transpose:
        print("Setting ignore_alphas=True for tranpose")
        args.ignore_alphas = True

    config = vars(args)
    config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")

    PCs, labels, num_labels = load_data(args.raw_dir, args.full)
    if os.environ.get("SMOKE_TEST"):
        PCs = [pc[: 60 + i] for i, pc in enumerate(PCs[:100])]
        labels = labels[:100]
        args.disable_wb = True

    wandb.init(
        project="pointcloud-net-k-fold",
        config=config,
        mode="disabled" if args.disable_wb else None,
    )

    input_dim = PCs[0].shape[0 if args.transpose else 1]
    hiponet = HiPoNet(
        input_dim,
        args.num_weights,
        args.threshold,
        args.K,
        args.J,
        args.device,
        args.sigma,
        ignore_alphas=args.transpose,
        use_geometric_laplacian=args.use_geometric_laplacian,
        diffusion_steps=args.diffusion_steps,
        use_attention=args.use_attention,
    )
    with torch.no_grad():
        # Create proper batch and mask for dimension inference
        pc_sample = PCs[0].to(args.device)
        if args.transpose:
            pc_sample = pc_sample.T
        # Add batch dimension: (N, d) -> (1, N, d)
        pc_sample = pc_sample.unsqueeze(0)
        # Create mask: all points are valid
        mask_sample = torch.ones((1, pc_sample.shape[1]), dtype=torch.bool, device=args.device)
        input_dim = hiponet(pc_sample, mask_sample).shape[1]
    mlp = MLP(input_dim, args.hidden_dim, num_labels, args.num_layers).to(args.device)
    
    train(hiponet, mlp, PCs, labels)

if __name__ == "__main__":
    main()