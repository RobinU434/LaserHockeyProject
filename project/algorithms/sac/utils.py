import torch
from torch import nn, Tensor

def get_min_q(q1: nn.Module, q2: nn.Module, s: Tensor, a: Tensor) -> Tensor:
    """get minimal q-value between two q networks

    Args:
        q1 (nn.Module): _description_
        q2 (nn.Module): _description_
        s (Tensor): (batch_dim, state_dim)
        a (Tensor): (batch_dim, action_dim)

    Returns:
        Tensor: min q value for each sample
    """
    q1_val = q1.forward(s, a)
    q2_val = q2.forward(s, a)
    q1_q2 = torch.cat([q1_val, q2_val], dim=1)
    min_q = torch.min(q1_q2, 1, keepdim=True)[0]
    return min_q