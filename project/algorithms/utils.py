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
