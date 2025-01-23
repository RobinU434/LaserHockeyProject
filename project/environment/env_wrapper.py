import numpy as np
from project.environment.hockey_env.hockey.hockey_env import HockeyEnv, Mode
from gymnasium import spaces

class SinglePlayerHockeyEnv(HockeyEnv):
    """Agent to train is always player 1

    Args:
        HockeyEnv (_type_): _description_
    """
    def __init__(
        self,
        opponent: object,  # the other agent
        keep_mode=True,
        mode=Mode.NORMAL,
        verbose=False,
    ):
        super().__init__(keep_mode, mode, verbose)
        self.opponent = opponent
        self.action_space = spaces.Box(-1, +1, (self.num_actions,), dtype=np.float32)


    def reset(self, opponent_starts: bool = None, mode = None, seed = None, options = None):
        """reset environment
        assume: opponent has always the second half of the action (player two)

        Args:
            opponent_starts (bool, optional): opponent has the puck first. Defaults to None.
            mode (_type_, optional): _description_. Defaults to None.
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if opponent_starts is not None:
            player_starts = not opponent_starts
        else:
            player_starts = opponent_starts         
        
        return super().reset(one_starting=player_starts, mode=mode, seed=seed, options=options)
    
    def step(self, action):
        obs_agent_two = self.opponent.self.obs_agent_two()