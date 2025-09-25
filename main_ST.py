import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from tqdm import tqdm
from utils.read_data import load_data_ST, load_data_ST_melanoma
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score
from torchinfo import summary

from models.graph_learning import HiPoNet, MLP
from argparse import ArgumentParser

import gc

gc.enable()

# Define the parameters using parser args
parser = ArgumentParser(description="Pointcloud net")
parser.add_argument("--dataset", type=str, default="dfci", help="Which ST dataset")
parser.add_argument(
    "--raw_dir", type=str, required=True, help="Directory where the raw data is stored"
)
parser.add_argument("--label_name", type=str, default="pTR_label", help="Label name")
parser.add_argument("--full", action="store_true")
parser.add_argument("--orthogonal", action="store_true")
parser.add_argument("--model", type=str, default="graph", help="Type of structure")
parser.add_argument("--task", type=str, default="AUC")
parser.add_argument("--num_weights", type=int, default=2, help="Number of weights")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument(
    "--sigma", type=float, default=0.5, help="Threshold for creating the graph"
)
parser.add_argument(
    "--spatial_threshold",
    type=float,
    default=0.5,
    help="Threshold for creating the graph",
)
parser.add_argument(
    "--gene_threshold", type=float, default=0.5, help="Threshold for creating the graph"
)
parser.add_argument(
    "--hidden_dim", type=int, default=500, help="Hidden dim for the MLP"
)
parser.add_argument("--num_layers", type=int, default=2, help="Number of MLP layers")
parser.add_argument("--lr", type=float, default=3e-4, help="Learnign Rate")
parser.add_argument("--wd", type=float, default=3e-3, help="Weight decay")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
parser.add_argument("--gpu", type=int, default=0, help="GPU index")
args = parser.parse_args()


wandb.init(project="pointcloud-net-spacegm-prediction", config=vars(args))

if args.gpu != -1 and torch.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"


def eval_roc_auc(model_spatial, model_gene, mlp, spaital_PCs, gene_PCs, labels, loader):
    model_spatial.eval()
    model_gene.eval()
    mlp.eval()
    pred = []
    with torch.no_grad():
        for idx in loader:
            X_spatial = model_spatial([spaital_PCs[i].to(args.device) for i in idx])
            X_gene = model_gene([gene_PCs[i].to(args.device) for i in idx])
            logits = mlp(torch.cat([X_spatial, X_gene], 1))
            preds = torch.argmax(logits, dim=1)
            pred.append(preds)
    pred = torch.cat(pred).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    return accuracy_score(labels, pred)


def train(model_spatial, model_gene, mlp, spaital_PCs, gene_PCs, labels):
    print(args)
    if args.task == "AUC":
        loss_fn = AUCMLoss(margin=1)
        opt = PESG(
            list(model_spatial.parameters())
            + list(model_gene.parameters())
            + list(mlp.parameters()),
            loss_fn=loss_fn,
            lr=args.lr,
            weight_decay=args.wd,
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(
            list(model_spatial.parameters())
            + list(model_gene.parameters())
            + list(mlp.parameters()),
            lr=args.lr,
            weight_decay=args.wd,
        )
    # if(args.raw_dir == 'dfci'):
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels.cpu().numpy()
    )
    # else:
    #     if(args.raw_dir == "charville"):
    #         split = [["c004"], ["c002"]]
    #         patient_c = torch.load(f"ST_preprocessed/patient_c_{args.raw_dir}_{args.label_name}.pt")
    #     elif(args.raw_dir == "upmc"):
    #         split = [["c006", "c002"], ["c003", "c004"]]
    #         patient_c = torch.load(f"ST_preprocessed/patient_c_{args.raw_dir}_{args.label_name}.pt")
    #     train_idx = []
    #     test_idx = []
    #     for idx in range(len(patient_c)):
    #         if(patient_c[idx] in split[args.fold]):
    #             test_idx.append(idx)
    #         else:
    #             train_idx.append(idx)
    train_idx = torch.LongTensor(train_idx).to(args.device)
    test_idx = torch.LongTensor(test_idx).to(args.device)
    train_loader = DataLoader(train_idx, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_idx, batch_size=args.batch_size, shuffle=False)
    labels = labels.to(args.device)
    best_aoc = eval_roc_auc(
        model_spatial,
        model_gene,
        mlp,
        spaital_PCs,
        gene_PCs,
        labels[test_idx],
        test_loader,
    )
    with tqdm(range(args.num_epochs)) as tq:
        for e, epoch in enumerate(tq):
            t_loss = 0
            preds = []
            model_spatial.train()
            model_gene.train()
            mlp.train()
            for idx in train_loader:
                opt.zero_grad()

                X_spatial = model_spatial([spaital_PCs[i].to(args.device) for i in idx])
                X_gene = model_gene([gene_PCs[i].to(args.device) for i in idx])
                logits = mlp(torch.cat([X_spatial, X_gene], 1))
                # import pdb; pdb.set_trace()
                preds.append(torch.argmax(logits, dim=1))
                loss = loss_fn(logits, labels[idx])  # *10
                loss.backward()
                opt.step()

                t_loss += loss.item()
                del (X_spatial, X_gene, logits, loss)
                torch.cuda.empty_cache()
                gc.collect()
            preds = torch.cat(preds).cpu().detach().numpy()
            train_aoc = roc_auc_score(
                labels[train_idx].cpu().numpy(), preds, average="micro"
            )
            test_aoc = eval_roc_auc(
                model_spatial,
                model_gene,
                mlp,
                spaital_PCs,
                gene_PCs,
                labels[test_idx],
                test_loader,
            )
            if test_aoc > best_aoc:
                best_aoc = test_aoc
                model_path = (
                    f"space_gm_model/model_{args.raw_dir}_{args.label_name}.pth"
                )

            tq.set_description(
                "Loss = %.4f, Train AOC = %.4f, Test AOC = %.4f, Best AOC = %.4f"
                % (t_loss, train_aoc.item(), test_aoc.item(), best_aoc)
            )
    print(f"Best AOC : {best_aoc}")


if __name__ == "__main__":
    spaital_PCs, gene_PCs, labels, num_labels = load_data_ST(
        args.raw_dir, args.dataset, args.label_name
    )
    # spaital_PCs, gene_PCs, labels, num_labels = load_data_ST_melanoma(args.raw_dir)
    model_spatial = (
        HiPoNet(
            dimension=spaital_PCs[0].shape[1],
            num_weights=1,
            threshold=args.spatial_threshold,
            K=args.K,
            device=args.device,
            sigma=args.sigma,
        )
        .to(args.device)
        .float()
    )
    model_gene = (
        HiPoNet(
            dimension=gene_PCs[0].shape[1],
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
            model_spatial([spaital_PCs[0][:5].to(args.device)]).shape[1]
            + model_gene([gene_PCs[0][:5].to(args.device)]).shape[1]
        )
    mlp = MLP(input_dim, args.hidden_dim, num_labels, args.num_layers).to(args.device)
    model_path = f"space_gm_model/model_{args.raw_dir}_{args.label_name}.pth"

    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'mlp_state_dict': mlp.state_dict(),
    #     'best_aoc': 0,
    #     'args': args
    # }, model_path)

    train(model_spatial, model_gene, mlp, spaital_PCs, gene_PCs, labels)
