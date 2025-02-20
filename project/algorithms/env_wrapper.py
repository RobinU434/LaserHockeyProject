import math

import numpy as np
from gymnasium import ActionWrapper, RewardWrapper
from gymnasium.spaces import Box, Discrete, MultiDiscrete


class TanhWrapper(RewardWrapper):
    def __init__(self, env, scan_steps: int = None):
        super().__init__(env)
        self.scan_steps = scan_steps
        self.counter = 0

        self.max_reward = -math.inf

    def _update_max(self, reward):
        if self.scan_steps is None or self.counter < self.scan_steps:
            self.max_reward = max(abs(reward), self.max_reward)
            self.counter += 1

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._update_max(reward)
        return observation, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        return math.tanh(reward / self.max_reward)


class SymLogWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        absolute = abs(reward)
        return reward / absolute * math.log(absolute + 1)


class Box2DiscreteActionWrapper(ActionWrapper):
    def __init__(self, env, n_actions: int):
        super().__init__(env)

        # TODO: make this more flexible and also accepts not only one dimensional Boxes...
        # TODO: use an embedding of Box2MultidiscreteActionWrapper and MD2DiscreteActionWrapper
        assert isinstance(
            self.env.action_space, Box
        ), "given env action space has to be continuous"
        assert (
            len(self.env.action_space.shape) == 1
        ), "supports only vectorized action space"
        assert self.env.action_space.shape == (
            1,
        ), "supports only one dimensional action space"

        self.n_actions = n_actions

        lower = self.env.action_space.low
        upper = self.env.action_space.high

        self.interpolation = np.linspace(lower, upper, self.n_actions)
        self.action_space = Discrete(self.n_actions, seed=self.env.action_space.seed())

    def action(self, action: int) -> np.ndarray:
        """get index and return continuous action

        Args:
            action (int): index of continuous action

        Returns:
            np.ndarray:
        """
        assert self.action_space.contains(
            action
        ), f"action: {action} not part of action space {self.action_space}"
        return self.interpolation[action]

    def get_continuous_actions(self):
        return self.interpolation


class Box2MultiDiscreteActionWrapper(ActionWrapper):
    def __init__(self, env, nvec: int | np.ndarray):
        super().__init__(env)
        assert isinstance(
            self.env.action_space, Box
        ), "given env action space has to be continuous"
        assert (
            len(self.env.action_space.shape) == 1
        ), "supports only vectorized action space"
        assert (
            self.env.action_space.shape[0] > 1
        ), "supports only one dimensional action space"

        assert (nvec > 1).all(), "more than one action interpolation expected"
        assert (
            nvec.shape == self.env.action_space.shape
        ), "action space and nvec have to have the same shape"
        self.action_dim = self.env.action_space.shape[0]

        self.n_vec = nvec

        self.cont_lower = self.env.action_space.low
        self.cont_upper = self.env.action_space.high
        self.span = self.cont_upper - self.cont_lower
        self.action_space = MultiDiscrete(self.n_vec, seed=self.env.action_space.seed())

    def action(self, action: np.ndarray) -> np.ndarray:
        """expect array of indices and returns the continuous version of it.

        Args:
            action (np.ndarray): ndarray of indices

        Returns:
            np.ndarray: vector of continuous actions
        """
        assert self.action_space.contains(
            action
        ), f"action: {action} not part of action space {self.action_space}"
        continuous_action = self.cont_lower + action / (self.n_vec - 1) * self.span
        return continuous_action

    def get_continuous_actions(self):
        res = [
            np.linspace(self.cont_lower[idx], self.cont_upper[idx], self.n_vec[idx])
            for idx in range(self.action_dim)
        ]
        return res


class MD2DiscreteActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        action_space: MultiDiscrete = env.action_space
        self.nvec = action_space.nvec
        self.num_actions = np.prod(self.nvec)
        self.action_space = Discrete(self.num_actions)

    def action(self, action):
        """Converts a discrete action index to a MultiDiscrete action tuple."""
        return np.array(np.unravel_index(action, self.nvec), dtype=np.int32)


class AffineActionTransform(ActionWrapper):
    def __init__(self, env, action_scale: np.ndarray, action_bias: np.ndarray):
        super().__init__(env)

        assert action_bias.shape == env.action_space.shape
        assert action_scale.shape == env.action_space.shape
        self.action_scale = action_scale
        self.action_bias = action_bias

    def action(self, action):
        return action * self.action_scale + self.action_bias
