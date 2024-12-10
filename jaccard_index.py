import torch
from torch import Tensor


def jaccard_index(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Jaccard index for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - inter

    jaccard = (inter + epsilon) / (union + epsilon)
    return jaccard.mean()


def jaccard_loss(input: Tensor, target: Tensor):
    # Jaccard loss (objective to minimize) between 0 and 1
    fn = jaccard_index
    return 1 - fn(input, target, reduce_batch_first=True)
