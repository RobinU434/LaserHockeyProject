from gymnasium import Space

from project.algorithms.common.agent import _Agent
from project.environment.hockey_env.hockey.hockey_env import BasicOpponent


class EvalOpponent(_Agent):
    def __init__(
        self, weak: bool = True, keep_mode: bool = True, verbose: bool = False
    ):
        super().__init__()
        self.verbose = verbose
        self.opponent = BasicOpponent(weak, keep_mode)

    def act(self, state):
        return self.opponent.act(state, self.verbose)


class RandomOpponent(_Agent):
    def __init__(self, action_space: Space):
        super().__init__()
        self.action_space = action_space

    def act(self, state):
        return self.action_space.sample()

