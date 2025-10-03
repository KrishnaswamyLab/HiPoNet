import torch
import pathlib


def collate_fn(batch):
    """Pad all the point clouds to have the same size."""
    input_tensor = torch.nested.as_nested_tensor(
        [x[0] for x in batch], layout=torch.jagged
    ).to_padded_tensor(padding=0.0)
    mask = input_tensor.sum(-1) != 0
    labels = torch.LongTensor([x[1] for x in batch])

    return input_tensor, mask, labels


def save_model(model: torch.nn.Module, name: str, location: pathlib.Path):
    torch.save(model.state_dict(), location / f"{name}.pt")
