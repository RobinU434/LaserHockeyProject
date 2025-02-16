from typing import Any, Dict
from gymnasium import Env, Space
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np


def get_space_dim(space: Space) -> int:
    shape = space.shape
    if len(shape) == 0:
        return 1
    return shape[0]


class ContinuousPlaceHolderEnv(Env):
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


class DiscretePlaceHolderEnv(Env):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()

        self.action_space = Discrete(n_actions)
        self.observation_space = Box(-1, 1, shape=(state_dim,))


class MulitDiscretePlaceHolderEnv(Env):
    def __init__(self, state_dim: int, nvec: np.ndarray):
        super().__init__()

        self.action_space = MultiDiscrete(nvec)
        self.observation_space = Box(-1, 1, shape=(state_dim,))


def build_placeholder_env(ckpt_content: Dict[str, Any]) -> Env:
    assert (
        "action_space" in ckpt_content.keys()
    ), "type of action space has to be defined"
    action_space_name = ckpt_content["action_space"]
    if action_space_name == Box.__name__:
        env = ContinuousPlaceHolderEnv(
            ckpt_content["state_dim"],
            ckpt_content["action_dim"],
            ckpt_content["action_scale"],
            ckpt_content["action_bias"],
        )
    elif action_space_name == Discrete.__name__:
        env = DiscretePlaceHolderEnv(
            ckpt_content["state_dim"],
            ckpt_content["n_actions"],
        )
    elif action_space_name == MultiDiscrete.__name__:
        env = MulitDiscretePlaceHolderEnv(
            ckpt_content["state_dim"],
            ckpt_content["nvec"],
        )
    else:
        raise ValueError(
            f"No Placeholder env for action space name: {action_space_name}. You can use a predefined env to not create an env"
        )

    return env
