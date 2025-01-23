from project.hockey_env.hockey.hockey_env import HockeyEnv, Mode


class SinglePlayerHockeyEnv(HockeyEnv):
    def __init__(
        self,
        opponent: object,  # the other agent
        keep_mode=True,
        mode=Mode.NORMAL,
        verbose=False,
    ):
        super().__init__(keep_mode, mode, verbose)
        self.opponent = opponent


    def reset(self, opponent_starts: bool = None, mode = None, seed = None, options = None):
        if opponent_starts is not None:
            opponent_starts = opponent_starts
        else:
            opponent_starts = not opponent_starts
        
        if opponent_starts:
            return super().reset(one_starting=True, mode=mode, seed=seed, options=options)
            