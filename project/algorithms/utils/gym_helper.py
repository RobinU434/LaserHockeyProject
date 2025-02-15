from gymnasium import Env, Space
from gymnasium.spaces import Box


def get_space_dim(space: Space) -> int:
    shape = space.shape
    if len(shape) == 0:
        return 1
    return shape[0]


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
        self.action_space = Box(lower_bound, upper_bound, shape=(action_dim,))

        self.observation_space = Box(-1, 1, shape=(state_dim,))
