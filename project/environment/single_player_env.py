import numpy as np
from gymnasium import spaces

from project.algorithms.agent import _Agent
from project.environment.hockey_env.hockey.hockey_env import HockeyEnv, Mode
from project.environment.opponent import RandomOpponent


class SinglePlayerHockeyEnv(HockeyEnv):
    """Agent to train is always player 1

    Args:
        HockeyEnv (_type_): _description_
    """

    def __init__(
        self,
        opponent: _Agent = None,  # the other agent
        keep_mode=True,
        mode=Mode.NORMAL,
        verbose=False,
    ):
        super().__init__(keep_mode, mode, verbose)
        self.action_space = spaces.Box(-1, +1, (self.num_actions,), dtype=np.float32)
        self.opponent = (
            opponent if opponent is not None else RandomOpponent(self.action_space)
        )

    def reset(self, opponent_starts: bool = None, mode=None, seed=None, options=None):
        """reset environment
        assume: opponent has always the second half of the action (player two)

        Args:
            opponent_starts (bool, optional): opponent has the puck first. Defaults to None.
            mode (_type_, optional): _description_. Defaults to None.
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[np.ndarray, dict]: state, info
        """
        if opponent_starts is not None:
            player_starts = not opponent_starts
        else:
            player_starts = opponent_starts

        return super().reset(
            one_starting=player_starts, mode=mode, seed=seed, options=options
        )

    def step(self, action: np.ndarray):
        obs_agent_two = self.obs_agent_two()
        action_agent_two = self.opponent.act(obs_agent_two)
        env_action = np.concat([action, action_agent_two])
        return super().step(env_action)
