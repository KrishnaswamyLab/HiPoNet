import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb
from utils.read_data import load_data

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
parser.add_argument(
    "--hidden_dim", type=int, default=256, help="Hidden dim for the MLP"
)
parser.add_argument("--num_layers", type=int, default=3, help="Number of MLP layers")
parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
parser.add_argument("--wd", type=float, default=3e-3, help="Weight decay")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--gpu", type=int, default=0, help="GPU index")
args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    print(f"Using {args.gpu} GPUs")
    args.device = "cuda"
else:
    args.device = "cpu"


def test(model, mlp, PCs, labels, loader):
    model.eval()
    mlp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx in loader:
            # X = model(idx, 0.000001)
            X = model([PCs[i].to(args.device) for i in idx], args.sigma)
            logits = mlp(X)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels[idx]).float()
            total += len(idx)
    return (correct * 100) / total


def train(model, mlp, PCs, labels):
    print(args)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(mlp.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2)
    train_idx = torch.LongTensor(train_idx).to(args.device)
    test_idx = torch.LongTensor(test_idx).to(args.device)
    train_loader = DataLoader(train_idx, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_idx, batch_size=args.batch_size)
    # labels = torch.LongTensor(model.labels).to(args.device)
    labels = torch.LongTensor(labels).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = test(model, mlp, PCs, labels, test_loader)
    with tqdm(range(args.num_epochs)) as tq:
        for e, epoch in enumerate(tq):
            correct_train = 0
            t_loss = 0
            model.train()
            mlp.train()
            for idx in train_loader:
                opt.zero_grad()

                X = model([PCs[i].to(args.device) for i in idx], args.sigma)
                logits = mlp(X)
                preds = torch.argmax(logits, dim=1)
                correct_train += torch.sum(preds == labels[idx]).float()
                loss = (
                    loss_fn(logits, labels[idx]) * 100
                )  # + 0.1*(model.layer.alphas@model.layer.alphas.T - torch.eye(args.num_weights).to(args.device)).square().mean()
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch + 1)
                for name, param in mlp.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch + 1)
                opt.step()
                t_loss += loss.item()
                del (X, logits, loss, preds)
                torch.cuda.empty_cache()
                gc.collect()

            train_acc = correct_train * 100 / len(train_idx)
            test_acc = test(model, mlp, PCs, labels, test_loader)
            wandb.log(
                {
                    "Loss": t_loss,
                    "Train acc": train_acc.item(),
                    "Test acc": test_acc.item(),
                },
                step=epoch + 1,
            )
            # for k in range(len(model.layer.alphas)):
            #     for d in range(len(model.layer.alphas[k])):
            #         wandb.log({f'Alpha{k}_{d}':model.layer.alphas[k][d].item()}, step=epoch+1)
            if test_acc > best_acc:
                best_acc = test_acc
                model_path = (
                    args.raw_dir + f"/simplex_models/model_{args.num_weights}.pth"
                )

                # torch.save({
                #     'epoch': epoch,  # Save the current epoch number
                #     'model_state_dict': model.state_dict(),
                #     'mlp_state_dict': mlp.state_dict(),
                #     'optimizer_state_dict': opt.state_dict(),
                #     'best_acc': best_acc,
                #     'args': args
                # }, model_path)

            tq.set_description(
                "Train Loss = %.4f, Train acc = %.4f, Test acc = %.4f, Best acc = %.4f"
                % (t_loss, train_acc.item(), test_acc.item(), best_acc)
            )
    print(f"Best accuracy : {best_acc}")


if __name__ == "__main__":
    wandb.init(project="pointcloud-net-k-fold", config=vars(args))

    PCs, labels, num_labels = load_data(args.raw_dir, args.full)
    model = HiPoNet(
        PCs[0].shape[1], args.num_weights, args.threshold, args.K, args.device
    )
    model = nn.DataParallel(model).to(args.device)
    with torch.no_grad():
        input_dim = model([PCs[0].to(args.device)], 1).shape[1]
    mlp = MLP(input_dim, args.hidden_dim, num_labels, args.num_layers).to(args.device)
    model_path = f"saved_models/model_{args.raw_dir}_{args.num_weights}_persistence_prediction.pth"

    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'mlp_state_dict': mlp.state_dict(),
    #     'best_acc': 0,
    #     'args': args
    # }, model_path)

    train(model, mlp, PCs, labels)
