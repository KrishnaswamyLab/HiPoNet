import torch


def collate_fn(batch):
    # We do this because DataParallel requires an explicit batch dimension
    # So we need our data to be a tensor, but the different point clouds have different
    # number of points, so we use nested tensors
    # We also want to ensure each GPU gets a similar amount of data
    # So we split the batch into segments which should have roughly even amount of data.
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return torch.nested.as_nested_tensor(
            [x[0] for x in batch], layout=torch.jagged
        ), torch.LongTensor([x[1] for x in batch])

    # We have n_gpus buckets, try and fill them up evenly
    max_per_bucket = len(batch) // n_gpus
    buckets = [[] for _ in range(n_gpus)]
    scores = [0 for _ in range(n_gpus)]
    for x in batch:
        min_idx = scores.index(min(scores))
        buckets[min_idx].append(x)
        if len(buckets[min_idx]) == max_per_bucket:
            # Bucket is full, set score to infinity so we don't add any more
            scores[min_idx] = float("inf")
        else:
            # The 'score' is the number of points *squared* since the W matrix has N^2 entries
            scores[min_idx] += x[0].shape[0] ** 2

    data = torch.nested.as_nested_tensor(
        [x[0] for bucket in buckets for x in bucket], layout=torch.jagged
    )
    labels = torch.LongTensor([x[1] for bucket in buckets for x in bucket])
    return data, labels
