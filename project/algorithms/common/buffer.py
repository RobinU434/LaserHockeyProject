from typing import Iterable, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np


class _ReplayBuffer(Dataset):
    def __init__(self, buffer_limit: int):
        super().__init__()
        self._buffer_limit = buffer_limit
        self._count = 0

        self._observations: Iterable
        self._actions: Iterable
        self._next_observations: Iterable
        self._rewards: Iterable
        self._dones: Iterable

    def put(
        self,
        observation: Tensor,
        action: Tensor,
        next_observation: Tensor,
        reward: Tensor,
        done: Tensor,
    ):
        """appends or extends buffer elements

        Args:
            observation (Tensor): (observation_shape,) | (n_samples, observation_shape)
            action (Tensor): (action_shape,) | (n_samples, action_shape)
            next_observation (Tensor): (observation_shape,) | (n_samples, observation_shape)
            reward (Tensor): (1,) | (n_samples, 1)
            done (Tensor): (1,) | (n_samples, 1)
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """_summary_

        Args:
            index (int): _description_

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: observations, actions, next_observations, rewards, dones
        """
        return (
            self._observations[index],
            self._actions[index],
            self._next_observations[index],
            self._rewards[index],
            self._dones[index],
        )

    def sample(
        self, batch_size: int = 1, replace: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """_summary_

        Args:
            index (int): _description_

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: observations, actions, next_observations, rewards, dones
        """
        sample_idx = np.random.choice(self._count, size=batch_size, replace=replace)
        return self[sample_idx]

    def __len__(self) -> int:
        return self._count


class ReplayBuffer(_ReplayBuffer):
    def __init__(self, buffer_limit):
        super().__init__(buffer_limit)
        self._actions: List = []
        self._observations: List = []
        self._next_observations: List = []
        self._rewards: List = []
        self._dones: List = []

    def put(self, observation, action, next_observation, reward, done):
        if len(action.shape) == 2:
            assert (
                action.shape[0] == observation.shape[0]
                and action.shape[0] == reward.shape[0]
            ), "If you add multiple actions, observations and rewards at once then add the same amount of each one"
            batch_size = len(action)
            if self._count + batch_size > self._buffer_limit:
                # remove overflow buffer element
                rm_slice = slice(0, self._count + batch_size - self._buffer_limit)
                self.pop_slice(rm_slice)

            self._actions.extend(list(action))
            self._observations.extend(list(observation))
            self._next_observations.extend(list(next_observation))
            self._rewards.extend(list(reward))
            self._dones.extend(list(done))
            return

        if self._count == self._buffer_limit:
            self.pop(0)
        self._actions.append(action)
        self._observations.append(observation)
        self._next_observations.append(next_observation)
        self._rewards.append(reward)
        self._dones.append(reward)
        self._count += 1

    def pop(self, index: int):
        self._actions.pop(index)
        self._observations.pop(index)
        self._next_observations.pop(index)
        self._rewards.pop(index)
        self._dones.pop(index)
        self._count -= 1

    def pop_slice(self, s: slice):
        del self._actions[s]
        del self._observations[s]
        del self._next_observations[s]
        del self._rewards[s]
        del self._dones[s]
        self._count = len(self._actions)


class RigidReplayBuffer(_ReplayBuffer):
    def __init__(
        self,
        buffer_limit: int,
        action_shape: torch.Size | List[int] | int,
        observation_shape: torch.Size | List[int] | int,
        reward_shape: torch.Size | List[int] | int,
    ):
        super().__init__(buffer_limit)
        self._actions: Tensor = torch.empty((self._buffer_limit, action_shape))
        self._observations: Tensor = torch.empty(
            (self._buffer_limit, observation_shape)
        )
        self._next_observations: Tensor = torch.empty(
            (self._buffer_limit, observation_shape)
        )
        self._rewards: Tensor = torch.empty((self._buffer_limit, reward_shape))
        self._dones: Tensor = torch.empty((self._buffer_limit, 1))

    def put(self, observation, action, next_observation, reward, done):
        shift_idx = 1
        if len(action.shape) == 2:
            assert (
                action.shape[0] == observation.shape[0]
                and action.shape[0] == next_observation.shape[0]
                and action.shape[0] == reward.shape[0]
                and action.shape[0] == done.shape[0]
            ), "If you add multiple actions, observations and rewards at once then add the same amount of each one"
            shift_idx = action.shape[0]

        self._actions[:-shift_idx] = self._actions[shift_idx:].clone()
        self._actions[-shift_idx:] = action
        self._observations[:-shift_idx] = self._observations[shift_idx:].clone()
        self._observations[-shift_idx:] = observation
        self._next_observations[:-shift_idx] = self._next_observations[
            shift_idx:
        ].clone()
        self._next_observations[-shift_idx:] = next_observation
        self._rewards[:-shift_idx] = self._rewards[shift_idx:].clone()
        self._rewards[-shift_idx:] = reward
        self._dones[:-shift_idx] = self._dones[shift_idx:].clone()
        self._dones[-shift_idx:] = reward

        if self._count < self._buffer_limit:
            self._count += 1
