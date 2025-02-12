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


class DiscreteActionWrapper(ActionWrapper):
    def __init__(self, env, n_actions: int):
        super().__init__(env)
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
        return self.interpolation[[action]]
    
    def get_continuous_actions(self):
        return self.interpolation



class MultiDiscreteActionWrapper(ActionWrapper):
    def __init__(self, env, n_actions: int | np.ndarray):
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

        msg = "more than one action interpolation expected"
        if isinstance(n_actions, int):
            assert n_actions > 1, msg
        else:
            assert (n_actions > 1).all(), msg

        self.action_dim = self.env.action_space.shape[0]

        if isinstance(n_actions, int):
            n_vec = np.ones(self.action_dim) * n_actions
        elif isinstance(n_actions, np.ndarray):
            n_vec = n_actions
        self.n_vec = n_vec

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
    
    

