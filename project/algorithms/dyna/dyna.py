import math
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from project.algorithms.common.agent import _Agent
from project.algorithms.common.algorithm import _RLAlgorithm
from project.algorithms.common.buffer import ReplayBuffer
from project.algorithms.common.q_net import DiscreteQNet
from project.algorithms.dyna.env_model import EnvModel
from project.algorithms.dyna.q_net import QNet
from project.algorithms.utils import PlaceHolderEnv, get_space_dim
from gymnasium.spaces import Discrete
from torch import Tensor

class DynaQ(_RLAlgorithm):
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
        assert isinstance(
            self._env.action_space, Discrete
        ), "Dyna requires a discrete action space"

        self._batch_size = batch_size
        self._buffer_limit = buffer_limit
        self._start_buffer_size = start_buffer_size
        self._epsilon_decay = epsilon_decay  # decay epsilon greedy policy
        self._gamma = gamma  # discount factor 
        self._tau = tau  # soft update factor
        self._simulation_updates = simulation_updates
        self.save_hyperparmeters()

        self._state_dim = get_space_dim(self._env.observation_space)  # noqa: F821
        self._n_actions = self._env.action_space.n

        self.q_net = QNet(self._state_dim, self._n_actions)
        self.q_target = QNet(self._state_dim, self._n_actions)
        self.q_target.load_state_dict(self.q_net.state_dict().copy())

        # note n_actions = action dim because of one hot encoding -> easiest encoding but others are possible
        self.world_model = EnvModel(self._state_dim, self._n_actions)
        self.memory = ReplayBuffer(self._buffer_limit)

        self.total_steps: int

    def _get_epsilon(self, total_steps: int):
        return math.exp(-self._epsilon_decay * total_steps)

    def encode_action(self, action: int | Tensor) -> Tensor:
        """converts the discrete action into an action encoding

        Args:
            action (int | Tensor): discrete action. if iterable: (batch_size, ) and dtype = int

        Returns:
            Tensor: action encoding of with a specific action dimension
        """
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
        """epsilon greedy policy

        Args:
            state (np.ndarray): current state

        Returns:
            int: discrete action
        """
        # decay epsilon exponentially
        epsilon = self._get_epsilon(self.total_steps)
        if np.random.uniform() > epsilon:  # greedy
            action_log_probs = self.q_net.action_log_probs(state)
            action = torch.argmax(action_log_probs).item()
            log_prob = action_log_probs[action]
            return action, log_prob
        
        action = np.random.randint(0, self._n_actions)
        log_prob = -math.log(self._n_actions)
        return action, log_prob

    def simulation_training(self, episode_idx: int):
        metrics = []
        for _ in range(self._simulation_updates):
            state, action, _, _, _ = self.memory.sample(self._batch_size)

            action_enc = self.encode_action(action)
            with torch.no_grad():
                next_state_pred, reward_pred, done_pred = self.world_model.forward(state, action_enc)
            minibatch = state, action, next_state_pred, reward_pred, done_pred
            target = self.calculate_target(minibatch)
            self.q_net.train(minibatch, target)

            info = self.q_net.soft_update(self.q_target, self._tau)
            metrics.append(info)
        
        metrics = pd.DataFrame(metrics).mean().to_dict()
        self.log_dict(metrics, episode_idx, prefix=self.get_name() + "/sim_")

    def update_world_model(
        self,
        mini_batch: Tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor
        ],
        episode_idx: int,
    ):
        state, action, next_state, reward, done = mini_batch
        action = self.encode_action(action)
        info = self.world_model.train_net(state, action, next_state, reward, done)
        self.log_dict(info, episode_idx, prefix=self.get_name() + "/")

    def calculate_target(
        self,
        mini_batch: Tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor
        ],
    ):
        _, _, next_state, reward, done = mini_batch
        with torch.no_grad():
            q_next = self.q_target.forward(next_state)
            q_next_max = torch.max(q_next, dim=-1, keepdim=True)[0]

        # print(reward, self._gamma, q_next_max, done)
        q_target = reward + self._gamma * q_next_max * (1 - done)
        return q_target

    def update_q(
        self,
        mini_batch: Tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor
        ],
        episode_idx: int,
    ):
        target = self.calculate_target(mini_batch)
        info = self.q_net.train(mini_batch, target)
        self.log_dict(info, episode_idx, prefix=self.get_name() + "/")

        # make cautious update of target network
        self.q_net.soft_update(self.q_target, self._tau)

        
    def train(self, n_episodes, verbose=False):
        if isinstance(self._env, PlaceHolderEnv):
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
                next_state, reward, done, truncated, _ = self._env.step(action)

                self.memory.put(
                    observation=torch.from_numpy(state),
                    action=torch.tensor([action], dtype=int),
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
            self.log_scalar(self.get_name() + "/log_prob", log_probs / episode_steps, episode_idx)



    def save_checkpoint(self, episode_idx, path=None):
        return super().save_checkpoint(episode_idx, path)

    def load_checkpoint(self, checkpoint):
        return super().load_checkpoint(checkpoint)

    def get_agent(self, deterministic=True):
        return super().get_agent(deterministic)

class DynaQQ(DynaQ):
    pass

class MultiDiscreteDynaQ(DynaQ):
    pass

class MultiDiscreteDynaQQ(DynaQ):
    pass


class DynaAgent(_Agent):
    def __init__(self):
        super().__init__()

    def act(self, state):
        return super().act(state)