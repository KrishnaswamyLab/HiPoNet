import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from utils.read_data import load_data_persistence
import wandb

from models.graph_learning import HiPoNet, MLP
from models.threshold_selection import select_threshold
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore")

import gc
gc.enable()

# Define the parameters using parser args
parser = ArgumentParser(description="Pointcloud net — persistence prediction")
parser.add_argument('--raw_dir', type=str, default='melanoma_data_full', help="Directory where the raw data is stored")
parser.add_argument('--full', action='store_true')
parser.add_argument('--orthogonal', action='store_true')
parser.add_argument('--sparse', action='store_true', help="Add L1 sparsity loss on alphas")
parser.add_argument('--sparse_lambda', type=float, default=0.01, help="Weight for L1 sparsity loss")
parser.add_argument('--num_weights', type=int, default=2, help="Number of weights")
parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for creating the graph")
parser.add_argument('--use_kneedle', action='store_true', help="Auto-select threshold per fold via Kneedle algorithm")
parser.add_argument('--sigma', type=float, default=0.5, help="Bandwidth")
parser.add_argument('--K', type=int, default=1, help="Order of simplicial complex")
parser.add_argument('--J', type=int, default=3, help="Number of wavelet scales")
parser.add_argument('--hidden_dim', type=int, default=500, help="Hidden dim for the MLP")
parser.add_argument('--num_layers', type=int, default=3, help="Number of MLP layers")
parser.add_argument('--lr', type=float, default=1e-1, help="Learning Rate")
parser.add_argument('--wd', type=float, default=3e-3, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--gpu', type=int, default=0, help="GPU index")
parser.add_argument('--disable_wb', action='store_true', help="Disable wandb logging")
parser.add_argument('--use_geometric_laplacian', action='store_true',
                    help="Use metric-aware geometric Hodge Laplacian (requires K >= 2)")
parser.add_argument('--diffusion_steps', type=int, default=1,
                    help="Number of diffusion steps t for computing P^t")
parser.add_argument('--use_attention', action='store_true',
                    help="Use DeepSet attention pooling over simplices (K >= 2)")
args = parser.parse_args()


wandb.init(
    project='pointcloud-net-persistence-prediction',
    config=vars(args),
    mode="disabled" if args.disable_wb else None,
)

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

loss_fn = torch.nn.MSELoss()


def make_persistence_dataset(PCs, labels):
    """Wrap point clouds and labels into (tensor, label_row) pairs."""
    dataset = []
    for i in range(len(PCs)):
        dataset.append((PCs[i], labels[i]))
    return dataset


def collate_fn_regression(batch):
    """Pad point clouds and stack tensor labels (not scalar)."""
    lengths = [x[0].shape[0] for x in batch]
    input_tensor = torch.nested.as_nested_tensor(
        [x[0] for x in batch], layout=torch.jagged
    ).to_padded_tensor(padding=0.0)
    arange = torch.arange(input_tensor.shape[1])
    mask = arange.unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    labels = torch.stack([x[1] for x in batch])
    return input_tensor, mask, labels


def test(model, mlp, loader):
    model.eval()
    mlp.eval()
    mse = 0
    total = 0
    with torch.no_grad():
        for batch, mask, label_batch in loader:
            batch, mask = batch.to(args.device), mask.to(args.device)
            label_batch = label_batch.to(args.device).float()
            X = model(batch, mask)
            preds = mlp(X)
            mse += (loss_fn(preds, label_batch) * len(label_batch))
            total += len(label_batch)
    return mse * 1000 / total


def train(model, mlp, train_loader, test_loader, fold=0, step_offset=0):
    print(args)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(mlp.parameters()),
        lr=args.lr, weight_decay=args.wd,
    )
    prefix = f"fold{fold}/"
    for k in range(len(model.layer.alphas)):
        for d in range(len(model.layer.alphas[k])):
            wandb.log({f'{prefix}Alpha{k}_{d}': model.layer.alphas[k][d].item()}, step=step_offset)
    train_mse = test(model, mlp, train_loader)
    best_mse = test(model, mlp, test_loader)
    wandb.log({f'{prefix}Train MSE': train_mse.item(), f'{prefix}Test MSE': best_mse.item()}, step=step_offset)
    with tqdm(range(args.num_epochs)) as tq:
        for e, epoch in enumerate(tq):
            t_loss = 0
            model.train()
            mlp.train()
            for batch, mask, label_batch in train_loader:
                batch, mask = batch.to(args.device), mask.to(args.device)
                label_batch = label_batch.to(args.device).float()
                opt.zero_grad()
                X = model(batch, mask)
                logits = mlp(X)
                loss = loss_fn(logits, label_batch) * 1000
                if args.orthogonal:
                    loss += 0.1 * (
                        model.layer.alphas @ model.layer.alphas.T
                        - torch.eye(args.num_weights).to(args.device)
                    ).square().mean()
                if args.sparse:
                    loss += args.sparse_lambda * model.layer.alphas.abs().sum()
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"{prefix}{name}.grad": param.grad.norm()}, step=step_offset + epoch + 1)
                opt.step()
                t_loss += loss.item()
                del X, logits, loss
                torch.cuda.empty_cache()
                gc.collect()

            train_mse = test(model, mlp, train_loader)
            test_mse = test(model, mlp, test_loader)
            step = step_offset + epoch + 1
            wandb.log({
                f'{prefix}Loss': t_loss,
                f'{prefix}Train MSE': train_mse.item(),
                f'{prefix}Test MSE': test_mse.item(),
            }, step=step)
            for k in range(len(model.layer.alphas)):
                for d in range(len(model.layer.alphas[k])):
                    wandb.log({f'{prefix}Alpha{k}_{d}': model.layer.alphas[k][d].item()}, step=step)
            if test_mse < best_mse:
                best_mse = test_mse

            tq.set_description(
                "Train MSE = %.4f, Test MSE = %.4f, Best MSE = %.4f"
                % (train_mse.item(), test_mse.item(), best_mse)
            )
    print(f"Best MSE : {best_mse}")
    return best_mse


if __name__ == '__main__':
    PCs, labels, num_labels = load_data_persistence(args.raw_dir, args.full)
    dataset = make_persistence_dataset(PCs, labels)
    input_dim = PCs[0].shape[1]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_best_mses = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"\n{'='*50}\nFold {fold + 1}/5\n{'='*50}")

        train_data = Subset(dataset, train_idx)
        test_data = Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn_regression,
        )
        test_loader = DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn_regression,
        )

        # Determine threshold: Kneedle over training PCs, or fixed arg
        if args.use_kneedle:
            train_PCs = [PCs[i] for i in train_idx]
            thresholds = [
                select_threshold(pc.float(), sigma=args.sigma)[0]
                for pc in train_PCs
            ]
            threshold = float(np.median(thresholds))
            print(f"Kneedle threshold (median over {len(train_PCs)} train PCs): {threshold:.4f}")
        else:
            threshold = args.threshold

        model = HiPoNet(
            input_dim,
            args.num_weights,
            threshold,
            args.K,
            args.J,
            args.device,
            args.sigma,
            use_geometric_laplacian=args.use_geometric_laplacian,
            diffusion_steps=args.diffusion_steps,
            use_attention=args.use_attention,
        ).to(args.device)

        # Infer MLP input dimension from a forward pass
        with torch.no_grad():
            sample_pc = PCs[train_idx[0]].to(args.device).unsqueeze(0)
            n_pts = PCs[train_idx[0]].shape[0]
            sample_mask = torch.ones((1, n_pts), dtype=torch.bool, device=args.device)
            feat_dim = model(sample_pc, sample_mask).shape[1]

        mlp = MLP(feat_dim, args.hidden_dim, num_labels, args.num_layers).to(args.device)

        wandb.log({"fold": fold + 1, "threshold": threshold})
        step_offset = fold * args.num_epochs
        best_mse = train(model, mlp, train_loader, test_loader, fold=fold + 1, step_offset=step_offset)
        fold_best_mses.append(best_mse)
        print(f"Fold {fold + 1} best test MSE: {best_mse:.4f}")

        del model, mlp
        torch.cuda.empty_cache()
        gc.collect()

    mean_mse = np.mean(fold_best_mses)
    std_mse = np.std(fold_best_mses)
    print(f"\n5-Fold CV Results: MSE = {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"Per-fold: {[f'{m:.4f}' for m in fold_best_mses]}")
    wandb.log({"cv_mean_mse": mean_mse, "cv_std_mse": std_mse})
