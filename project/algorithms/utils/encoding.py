import numpy as np
import torch


def one_hot(x: torch.Tensor, n: int) -> torch.Tensor:
    """one hot encoding for

    Args:
        x (torch.Tensor): (batch_size, ) vector of indices
        n (int): max index

    Returns:
        torch.Tensor: binary vector (batch_size, n)
    """

    action = x.int()
    batch_size = len(action)
    res = torch.zeros(batch_size, n, dtype=float)
    res[torch.arange(batch_size, dtype=int), action] = 1
    res = res.float()
    return res


def multi_hot(x: torch.Tensor, nvec: torch.Tensor) -> torch.Tensor:
    """multihot encoding. Same

    Args:
        x (torch.Tensor): (batch_size, feature_dim) in index space
        nvec (torch.Tensor): max indices per feature dim (feature_dim, )

    Returns:
        torch.Tensor: binary vector (batch_size, sum(nvec))
    """

    batch_size, _ = x.shape
    output_size = nvec.sum().item()
    output = torch.zeros((batch_size, output_size), dtype=torch.int64)

    start_indices = torch.cumsum(torch.cat((torch.tensor([0]), nvec[:-1])), dim=0)

    indices = start_indices.unsqueeze(0) + x
    output.scatter_(1, indices, 1)

    return output
