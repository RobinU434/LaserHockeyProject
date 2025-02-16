from copy import deepcopy
import math
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from gymnasium.spaces import Discrete, MultiDiscrete
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from project.algorithms.common.agent import _Agent
from project.algorithms.common.algorithm import (
    _RLAlgorithm,
    _DiscreteRLAlgorithm,
    _MultiDiscreteRLAlgorithm,
)
from project.algorithms.common.buffer import ReplayBuffer
from project.algorithms.common.q_net import _QNet
from project.algorithms.dyna.q_net import MultiDiscreteQNet, QNet
from project.algorithms.dyna.world_model import WorldModel
from project.algorithms.utils.encoding import multi_hot
from project.algorithms.utils.filesystem import get_save_path
from project.algorithms.utils.gym_helper import ContinuousPlaceHolderEnv, get_space_dim
from project.algorithms.utils.str_ops import generate_separator
from project.algorithms.utils.torch_ops import state_dict_to_cpu


class _DynaQ(_RLAlgorithm):
    def __init__(
        self,
        env,
        logger=[],
        eval_env=None,
        eval_check_interval=None,
        save_interval=None,
        log_dir=None,
        batch_size: int = 256,
        buffer_limit: int = 50_000,
        start_buffer_size: int = 1000,
        epsilon_decay: float = 0.99,  # decay epsilon greedy policy
        gamma: float = 0.99,  # discount factor
        tau: float = 0.01,  # soft update parameter
        simulation_updates: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(
            env,
            logger,
            eval_env,
            eval_check_interval,
            save_interval,
            log_dir,
            *args,
            **kwargs,
        )

        self._batch_size = batch_size
        self._buffer_limit = buffer_limit
        self._start_buffer_size = start_buffer_size
        self._epsilon_decay = epsilon_decay  # decay epsilon greedy policy
        self._gamma = gamma  # discount factor
        self._tau = tau  # soft update factor
        self._simulation_updates = simulation_updates
        self.save_hyperparmeters()

        self._state_dim = get_space_dim(self._env.observation_space)  # noqa: F821

        self.memory = ReplayBuffer(self._buffer_limit)

        self.total_steps: int

        self.q_net: _QNet
        self.q_target: _QNet
        self.world_model: WorldModel

    def _get_epsilon(self, total_steps: int):
        return math.exp(-self._epsilon_decay * total_steps)

    def encode_action(self, action: int | Tensor) -> Tensor:
        """converts the discrete action into an action encoding

        Args:
            action (int | Tensor): discrete action. if iterable: (batch_size, ) and dtype = int

        Returns:
            Tensor: action encoding of with a specific action dimension
        """
        raise NotImplementedError

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """epsilon greedy policy

        Args:
            state (np.ndarray): current state

        Returns:
            int: discrete action
        """
        raise NotImplementedError

    def get_max_q(self, state: Tensor) -> Tensor:
        """get maximum q value of target network

        Args:
            state (Tensor): (batch_size, state_dim)

        Returns:
            Tensor: (batch_size, 1)
        """
        raise NotImplementedError

    def simulation_training(self, episode_idx: int):
        metrics = []
        for _ in range(self._simulation_updates):
            state, action, _, _, _ = self.memory.sample(self._batch_size)

            action_enc = self.encode_action(action)
            with torch.no_grad():
                next_state_pred, reward_pred, done_pred = self.world_model.forward(
                    state, action_enc
                )
            minibatch = state, action, next_state_pred, reward_pred, done_pred
            target = self.calculate_target(minibatch)
            self.q_net.train_net(minibatch, target)

            info = self.q_net.soft_update(self.q_target, self._tau)
            metrics.append(info)

        metrics = pd.DataFrame(metrics).mean().to_dict()
        self.log_dict(metrics, episode_idx, prefix=self.get_name() + "/sim_")

    def update_world_model(
        self,
        mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        episode_idx: int,
    ):
        state, action, next_state, reward, done = mini_batch
        action = self.encode_action(action)
        info = self.world_model.train_net(state, action, next_state, reward, done)
        self.log_dict(info, episode_idx, prefix=self.get_name() + "/")

    def calculate_target(
        self,
        mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> Tensor:
        """calculate TD target for q values

        Args:
            mini_batch (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): content:
                state: (batch_size, state_dim)
                action: (batch_size, action_dim) in the discrete case action_dim=1
                next_state: (batch_size, state_dim)
                reward: (batch_size, 1)
                done: (batch_size, 1)

        Returns:
            Tensor: td target. (batch_size, 1)
        """
        _, _, next_state, reward, done = mini_batch
        with torch.no_grad():
            q_next_max = self.get_max_q(next_state)

        q_target = reward + self._gamma * q_next_max * (1 - done)
        return q_target

    def update_q(
        self,
        mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        episode_idx: int,
    ):
        target = self.calculate_target(mini_batch)
        info = self.q_net.train_net(mini_batch, target)
        self.log_dict(info, episode_idx, prefix=self.get_name() + "/")

        # make cautious update of target network
        self.q_net.soft_update(self.q_target, self._tau)

    def train(self, n_episodes, verbose=False):
        if isinstance(self._env, ContinuousPlaceHolderEnv):
            raise ValueError(
                "Training with PlaceHolderEnv is not possible. Please update internal environment."
            )

        iterator = range(n_episodes)
        if verbose:
            iterator = tqdm(iterator, desc="train sac", unit="episodes")

        self.total_steps = 0
        for episode_idx in iterator:
            score = 0
            episode_steps = 0
            log_probs = 0

            done = False
            truncated = False
            state, _ = self._env.reset()
            while not (done or truncated):
                action, log_prob = self.get_action(torch.from_numpy(state))
                # Discrete vs MultiDiscrete
                next_state, reward, done, truncated, _ = self._env.step(action)
                if isinstance(action, int):
                    action = np.array([action], dtype=int)

                self.memory.put(
                    observation=torch.from_numpy(state),
                    action=torch.from_numpy(action),
                    next_observation=torch.from_numpy(next_state),
                    reward=torch.tensor([reward]),
                    done=torch.tensor([done], dtype=float),
                    sampling_weight=None,
                )

                if len(self.memory) > self._start_buffer_size:
                    mini_batch = self.memory.sample(self._batch_size)
                    self.update_world_model(mini_batch, episode_idx)
                    self.update_q(mini_batch, episode_idx)
                    self.simulation_training(episode_idx)

                score += reward
                episode_steps += 1
                log_probs += log_prob

                state = next_state
                self.total_steps += 1

            self.log_scalar("mean_score", score / episode_steps, episode_idx)
            self.log_scalar("episode_steps", episode_steps, episode_idx)
            self.log_scalar(
                self.get_name() + "/log_prob", log_probs / episode_steps, episode_idx
            )


class DynaQ(_DynaQ, _DiscreteRLAlgorithm):
    def __init__(
        self,
        env,
        logger=[],
        eval_env=None,
        eval_check_interval=None,
        save_interval=None,
        log_dir=None,
        batch_size: int = 256,
        buffer_limit: int = 50_000,
        start_buffer_size: int = 1000,
        epsilon_decay: float = 0.99,  # decay epsilon greedy policy
        gamma: float = 0.99,  # discount factor
        tau: float = 0.01,  # soft update parameter
        simulation_updates: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(
            env,
            logger,
            eval_env,
            eval_check_interval,
            save_interval,
            log_dir,
            batch_size,
            buffer_limit,
            start_buffer_size,
            epsilon_decay,
            gamma,
            tau,
            simulation_updates,
            *args,
            **kwargs,
        )
        # TODO: test is not good. only Discrete but this conflicts MultiDiscrete descendent
        assert isinstance(
            self._env.action_space, Discrete
        ), "Dyna requires a discrete action space"

        self._n_actions = self._env.action_space.n

        self.q_net: QNet = QNet(self._state_dim, self._n_actions)
        self.q_target: QNet = QNet(self._state_dim, self._n_actions)
        self.q_target.load_state_dict(self.q_net.state_dict().copy())

        # note n_actions = action dim because of one hot encoding -> easiest encoding but others are possible
        self.world_model = WorldModel(self._state_dim, self._n_actions)

    def get_max_q(self, state):
        q_value = self.q_target.complete_forward(state)
        q_max = torch.max(q_value, dim=-1, keepdim=True)[0]
        return q_max

    def encode_action(self, action: int | Tensor) -> Tensor:
        # NOTE: only one hot
        if isinstance(action, int):
            a = torch.zeros(self._n_actions)
            a[action] = 1
            a = a.float()
            return a

        action = action.int()
        batch_size = len(action)
        a = torch.zeros(batch_size, self._n_actions, dtype=float)
        a[torch.arange(batch_size, dtype=int), action] = 1
        a = a.float()
        return a

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        # decay epsilon exponentially
        epsilon = self._get_epsilon(self.total_steps)
        if np.random.uniform() <= epsilon:  # random action
            action = np.random.randint(0, self._n_actions)
            log_prob = -math.log(self._n_actions)
            return action, log_prob

        # greedy action
        action_log_probs = self.q_net.action_log_probs(state)
        action = torch.argmax(action_log_probs).item()
        log_prob = action_log_probs[action]
        return action, log_prob

    def save_checkpoint(self, episode_idx, path=None):
        path = get_save_path(self._log_dir, episode_idx, path)
        content = self.get_basic_save_args(episode_idx)
        content = {
            **content,
            "q_state_dict": state_dict_to_cpu(self.q_net.state_dict()),
            "q_optimizer_state_dict": state_dict_to_cpu(
                self.q_net.optimizer.state_dict()
            ),
            "q_target_state_dict": state_dict_to_cpu(self.q_target.state_dict()),
            "q_target_optimizer_state_dict": state_dict_to_cpu(
                self.q_target.optimizer.state_dict()
            ),
            "world_model_state_dict": state_dict_to_cpu(self.world_model.state_dict()),
            "world_model_optimizer_state_dict": state_dict_to_cpu(
                self.world_model.optimizer.state_dict()
            ),
        }
        torch.save(content, path)

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only=False)
        self.q_net.load_state_dict(checkpoint[""])
        self.q_net.load_state_dict(checkpoint["q_state_dict"])
        self.q_net.optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.q_target.load_state_dict(checkpoint["q_target_state_dict"])
        self.q_target.optimizer.load_state_dict(
            checkpoint["q_target_optimizer_state_dict"]
        )
        self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        self.world_model.optimizer.load_state_dict(
            checkpoint["world_model_optimizer_state_dict"]
        )

    def __repr__(self):
        s1 = generate_separator("Q-Function", 80)
        s2 = generate_separator("World Model", 80)
        q_str = str(self.q_net)
        wm_str = str(self.world_model)
        s = "\n".join([s1, q_str, s2, wm_str])
        return s

    def get_agent(self, deterministic=True):
        return DiscreteDynaQAgent(deepcopy(self.q_net), deterministic)


class MultiDiscreteDynaQ(_DynaQ, _MultiDiscreteRLAlgorithm):
    def __init__(
        self,
        env,
        logger=[],
        eval_env=None,
        eval_check_interval=None,
        save_interval=None,
        log_dir=None,
        batch_size=256,
        buffer_limit=50000,
        start_buffer_size=1000,
        epsilon_decay=0.99,
        gamma=0.99,
        tau=0.01,
        simulation_updates=2,
        mc_sample=10,  # switch it of by setting it None. Then do whole sweep
        *args,
        **kwargs,
    ):
        super().__init__(
            env,
            logger,
            eval_env,
            eval_check_interval,
            save_interval,
            log_dir,
            batch_size,
            buffer_limit,
            start_buffer_size,
            epsilon_decay,
            gamma,
            tau,
            simulation_updates,
            *args,
            **kwargs,
        )
        assert isinstance(
            self._env.action_space, MultiDiscrete
        ), "expected MultiDiscrete action space"

        self._batch_size = batch_size
        self._buffer_limit = buffer_limit
        self._start_buffer_size = start_buffer_size
        self._epsilon_decay = epsilon_decay  # decay epsilon greedy policy
        self._gamma = gamma  # discount factor
        self._tau = tau  # soft update factor
        self._simulation_updates = simulation_updates
        self._mc_sample = mc_sample
        self.save_hyperparmeters()

        self._state_dim = get_space_dim(self._env.observation_space)  # noqa: F821
        action_space: MultiDiscrete = self._env.action_space
        self._action_dim = action_space.nvec.sum()
        self._n_actions = action_space.nvec.prod()

        self.q_net: MultiDiscreteQNet = MultiDiscreteQNet(
            self._state_dim, action_space.nvec
        )
        self.q_target: MultiDiscreteQNet = MultiDiscreteQNet(
            self._state_dim, action_space.nvec
        )
        self.q_target.load_state_dict(self.q_net.state_dict().copy())

        self.world_model = WorldModel(self._state_dim, self._action_dim)

    def encode_action(self, action: Tensor) -> Tensor:
        """converts the discrete action into an action encoding

        Args:
            action (int | Tensor): discrete action in index space (batch_size, feature_dim).

        Returns:
            Tensor: action encoding of with a specific action dimension (batch_size, encoding_dim)
        """

        if len(action.shape) == 1:
            action = action[None]

        nvec = torch.from_numpy(self._env.action_space.nvec).int()
        action = multi_hot(action.int(), nvec).float()
        return action

    def get_max_q(self, state):
        if self._mc_sample is None:
            q_value, _ = self.q_target.complete_forward(state)
        else:
            q_value, _ = self.q_target.mc_forward(state, self._mc_sample)
        q_value = q_value[..., 0]
        q_max = torch.max(q_value, dim=-1, keepdim=True)[0]
        return q_max

    def get_action(self, state):
        epsilon = self._get_epsilon(self.total_steps)
        if np.random.uniform() <= epsilon:  # random action
            action_idx = np.random.randint(0, self._n_actions)
            action = self.q_net._all_actions[action_idx]
            log_prob = -math.log(self._n_actions)
            return action, log_prob

        # greedy action
        if self._mc_sample is None:
            q_val, actions = self.q_net.complete_forward(state)
        else:
            q_val, actions = self.q_net.mc_forward(state, self._mc_sample)
        q_val = q_val[0]
        q_val = F.log_softmax(q_val, dim=0)
        idx = q_val.argmax(dim=0)
        log_prob = q_val[idx]
        action = actions[idx]
        return action, log_prob

    def save_checkpoint(self, episode_idx, path=None):
        path = get_save_path(self._log_dir, episode_idx, path)
        content = self.get_basic_save_args(episode_idx)

        content = {
            **content,
            "q_state_dict": state_dict_to_cpu(self.q_net.state_dict()),
            "q_optimizer_state_dict": state_dict_to_cpu(
                self.q_net.optimizer.state_dict()
            ),
            "q_target_state_dict": state_dict_to_cpu(self.q_target.state_dict()),
            "q_target_optimizer_state_dict": state_dict_to_cpu(
                self.q_target.optimizer.state_dict()
            ),
            "world_model_state_dict": state_dict_to_cpu(self.world_model.state_dict()),
            "world_model_optimizer_state_dict": state_dict_to_cpu(
                self.world_model.optimizer.state_dict()
            ),
        }
        torch.save(content, path)

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only=False)
        self.q_net.load_state_dict(checkpoint[""])
        self.q_net.load_state_dict(checkpoint["q_state_dict"])
        self.q_net.optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.q_target.load_state_dict(checkpoint["q_target_state_dict"])
        self.q_target.optimizer.load_state_dict(
            checkpoint["q_target_optimizer_state_dict"]
        )
        self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        self.world_model.optimizer.load_state_dict(
            checkpoint["world_model_optimizer_state_dict"]
        )

    def __repr__(self):
        s1 = generate_separator("Q-Function", 80)
        s2 = generate_separator("World Model", 80)
        q_str = str(self.q_net)
        wm_str = str(self.world_model)
        s = "\n".join([s1, q_str, s2, wm_str])
        return s

    def get_agent(self, deterministic=True):
        return MultiDiscreteDynaQAgent(
            deepcopy(self.q_net), deterministic, self._mc_sample
        )


class DiscreteDynaQAgent(_Agent):
    def __init__(self, q_net: QNet, deterministic: bool = False):
        super().__init__()
        self.deterministic = deterministic
        self.q_net = q_net

    def act(self, state):
        with torch.no_grad():
            action_probs = self.q_net.action_probs(state)
        if self.deterministic:
            return torch.argmax(action_probs).item()
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
        return action


class MultiDiscreteDynaQAgent(_Agent):
    def __init__(
        self,
        q_net: MultiDiscreteQNet,
        deterministic: bool = False,
        mc_sample: int = None,
    ):
        super().__init__()
        self.q_net = q_net
        self.deterministic = deterministic
        self.mc_sample = mc_sample

    def act(self, state):
        with torch.no_grad():
            if self.mc_sample is None:
                # do full sweep
                q_val, actions = self.q_net.complete_forward(state)
            else:
                q_val, actions = self.q_net.mc_forward(state, self.mc_sample)

        q_val = q_val[0]
        actions = actions[0]

        if self.deterministic:
            idx = torch.argmax(q_val, dim=0)
            return actions[idx]

        p = torch.softmax(q_val, dim=0)[:, 0]
        idx = np.random.choice(len(q_val), p=p)
        return actions[idx]
