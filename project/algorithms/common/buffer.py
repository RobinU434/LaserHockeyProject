from collections import deque
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

        self._obs: Iterable[Tensor]
        self._actions: Iterable[Tensor]
        self._next_obs: Iterable[Tensor]
        self._rewards: Iterable[Tensor]
        self._dones: Iterable[Tensor]

        self._sampling_weights = Iterable[float]

    def put(
        self,
        observation: Tensor,
        action: Tensor,
        next_observation: Tensor,
        reward: Tensor,
        done: Tensor,
        sampling_weight: Tensor | float = None
    ):
        """appends or extends buffer elements

        Args:
            observation (Tensor): (observation_shape,) | (n_samples, observation_shape)
            action (Tensor): (action_shape,) | (n_samples, action_shape)
            next_observation (Tensor): (observation_shape,) | (n_samples, observation_shape)
            reward (Tensor): (1,) | (n_samples, 1)
            done (Tensor): (1,) | (n_samples, 1)
            sampling_weight (Tensor) | float:  (n_samples,) or simple float
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
            self._obs[index].detach().float(),
            self._actions[index].detach().float(),
            self._next_obs[index].detach().float(),
            self._rewards[index].detach().float(),
            self._dones[index].detach().float(),
        )
    
    def _get_sample_idx(self, n: int, replace: bool = False) -> np.ndarray:
        if None in self._sampling_weights:
            weights = None 
        else:
            weights = np.array(self._sampling_weights)
            weights /= weights.sum()

        return np.random.choice(self._count, size=n, replace=replace, p=weights)

    def sample(
        self, batch_size: int = 1, replace: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """_summary_

        Args:
            index (int): _description_

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: observations, actions, next_observations, rewards, dones
        """

        sample_idx = self._get_sample_idx(batch_size, replace)
        return self[sample_idx]

    def __len__(self) -> int:
        return self._count


class ReplayBuffer(_ReplayBuffer):
    def __init__(self, buffer_limit):
        super().__init__(buffer_limit)
        self._actions: deque = deque(maxlen=buffer_limit)
        self._obs: deque = deque(maxlen=buffer_limit)
        self._next_obs: deque = deque(maxlen=buffer_limit)
        self._rewards: deque = deque(maxlen=buffer_limit)
        self._dones: deque = deque(maxlen=buffer_limit)
        self._sampling_weights: deque = deque(maxlen=buffer_limit)

    def put(self, observation, action, next_observation, reward, done, sampling_weight = None
):
        # print(observation.shape, action.shape, next_observation.shape)
        if len(action.shape) == 2:
            assert (
                action.shape[0] == observation.shape[0]
                and action.shape[0] == reward.shape[0]
                and action.shape[0] == next_observation.shape[0]
                and action.shape[0] == done.shape[0]
            ), "If you add multiple actions, observations and rewards at once then add the same amount of each one"

            self._actions.extend(list(action))
            self._obs.extend(list(observation))
            self._next_obs.extend(list(next_observation))
            self._rewards.extend(list(reward))
            self._dones.extend(list(done))
            self._sampling_weights.extend(list(sampling_weight))

            return

        self._obs.append(observation)
        self._actions.append(action)
        self._next_obs.append(next_observation)
        self._rewards.append(reward)
        self._dones.append(done)
        self._sampling_weights.append(sampling_weight)

    def sample(self, batch_size=1, replace=False):
        sample_idx = np.random.choice(len(self), batch_size, replace=replace)
        actions_lst = []
        observations_lst = []
        next_observations_lst = []
        rewards_lst = []
        dones_lst = []

        for idx in sample_idx:
            observations_lst.append(self._obs[idx])
            actions_lst.append(self._actions[idx])
            next_observations_lst.append(self._next_obs[idx])
            rewards_lst.append(self._rewards[idx])
            dones_lst.append(self._dones[idx])

        observations_lst = torch.stack(observations_lst).detach().float()
        actions_lst = torch.stack(actions_lst).detach().float()
        next_observations_lst = torch.stack(next_observations_lst).detach().float()
        rewards_lst = torch.stack(rewards_lst).detach().float()
        dones_lst = torch.stack(dones_lst).detach().float()

        return (
            observations_lst,
            actions_lst,
            next_observations_lst,
            rewards_lst,
            dones_lst,
        )

    def __len__(self) -> int:
        return len(self._actions)


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
        self._obs: Tensor = torch.empty((self._buffer_limit, observation_shape))
        self._next_obs: Tensor = torch.empty((self._buffer_limit, observation_shape))
        self._rewards: Tensor = torch.empty((self._buffer_limit, reward_shape))
        self._dones: Tensor = torch.empty((self._buffer_limit, 1))
        self._sampling_weights: Tensor = torch.empty((self._buffer_limit, 1))

    def put(self, observation, action, next_observation, reward, done, sampling_weight = None):
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
        self._obs[:-shift_idx] = self._obs[shift_idx:].clone()
        self._obs[-shift_idx:] = observation
        self._next_obs[:-shift_idx] = self._next_obs[shift_idx:].clone()
        self._next_obs[-shift_idx:] = next_observation
        self._rewards[:-shift_idx] = self._rewards[shift_idx:].clone()
        self._rewards[-shift_idx:] = reward
        self._dones[:-shift_idx] = self._dones[shift_idx:].clone()
        self._dones[-shift_idx:] = done
        self._sampling_weights[:-shift_idx] = self._sampling_weights[shift_idx:].clone()
        self._sampling_weights[-shift_idx:] = sampling_weight
        

        if self._count < self._buffer_limit:
            self._count += shift_idx
