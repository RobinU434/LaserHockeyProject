from gymnasium import Env, Space
from gymnasium.spaces import Box


def get_space_dim(space: Space) -> int:
    shape = space.shape
    if len(shape) == 0:
        return 1
    return shape[0]


class PlaceHolderEnv(Env):
    def __init__(self, state_dim: int, action_dim: int):
        """NOTE: assumes continuous action and state spaces

        Args:
            state_dim (int): _description_
            action_dim (int): _description_

        """
        self.observation_space = Box(-1, 1, shape=(state_dim,))
        self.action_space = Box(-1, 1, shape=(action_dim,))
