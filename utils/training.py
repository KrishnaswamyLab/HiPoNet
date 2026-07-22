import torch
import pathlib


def collate_fn(batch, transpose=False):
    """Pad all the point clouds to have the same size."""
    if transpose:
        input_tensor = torch.nested.as_nested_tensor(
            [x[0].T for x in batch], layout=torch.jagged
        )
        # All true (no need for mask since nested tensor)
        mask = torch.ones(input_tensor.shape[:-1], dtype=torch.bool)
    else:
        lengths = [x[0].shape[0] for x in batch]
        input_tensor = torch.nested.as_nested_tensor(
            [x[0] for x in batch], layout=torch.jagged
        ).to_padded_tensor(padding=0.0)
        arange = torch.arange(input_tensor.shape[1])
        mask = arange.unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    labels = torch.LongTensor([x[1] for x in batch])

    return input_tensor, mask, labels


def save_model(model: torch.nn.Module, name: str, location: pathlib.Path):
    torch.save(model.state_dict(), location / f"{name}.pt")
