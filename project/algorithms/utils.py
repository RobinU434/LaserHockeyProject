from gymnasium import Env, Space
from gymnasium.spaces import Box
import torch
from torch import nn, Tensor


def get_space_dim(space: Space) -> int:
    shape = space.shape
    if len(shape) == 0:
        return 1
    return shape[0]


def detach_tensor(*args):
    res = []
    for t in args:
        t.detach()
        res.append(t)
    return res


class PlaceHolderEnv(Env):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 2,
        action_bias: float = 0,
    ):
        """NOTE: assumes continuous action and state spaces

        Args:
            state_dim (int): _description_
            action_dim (int): _description_
            action_scale (float):

        """
        if action_scale is None:
            action_scale = 2

        lower_bound = action_bias - action_scale / 2
        upper_bound = action_bias + action_scale / 2
        self.observation_space = Box(lower_bound, upper_bound, shape=(state_dim,))
        self.action_space = Box(-1, 1, shape=(action_dim,))


def generate_separator(content: str, width: int, fill_char: str = "="):
    assert (
        len(content) + 2 < width
    ), f"Content: {content} + 2 white space has to be bigger than desired with"
    content = " " + content.strip(" ") + " "
    fill_len = (width - len(content)) // 2
    content = fill_char * fill_len + content
    content = content + fill_char * (width - len(content))
    return content


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
