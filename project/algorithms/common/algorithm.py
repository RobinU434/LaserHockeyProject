import inspect
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, List, Dict

import gymnasium
import torch
from torch import nn
from gymnasium import Env

from project.algorithms.common.agent import _Agent
from project.algorithms.common.logger import _Logger
from project.algorithms.utils.gym_helper import build_placeholder_env
from project.environment.evaluate_env import _EvalEnv
from gymnasium.spaces import Box, Discrete, MultiDiscrete


class _RLAlgorithm(ABC):
    def __init__(
        self,
        env: Env,
        logger: List[_Logger] = [],
        eval_env: List[_EvalEnv] | _EvalEnv = None,
        eval_check_interval: int = None,
        save_interval: int = None,
        log_dir: Path = None,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__()

        self._env = env
        self._logger = logger
        self._eval_envs = eval_env if isinstance(eval_env, list) else [eval_env]
        self._eval_check_interval = eval_check_interval
        self._save_interval = save_interval
        self._log_dir = (
            log_dir if log_dir is not None else Path().cwd().joinpath("results/")
        )
        self._log_dir = (
            Path(self._log_dir)
            if not isinstance(self._log_dir, Path)
            else self._log_dir
        )
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        self.hparams = Namespace()
        self.episode_offset = 0

    def save_hyperparmeters(self, *args):
        # Get the frame of the calling function (i.e., the __init__ method)
        frame = inspect.currentframe().f_back

        # Get the arguments of the calling function
        arg_values = frame.f_locals

        # pre filter
        exclude_args = [
            "self",
            "env",
            "logger",
            "eval_env",
            "kwargs",
            "args",
            "__class__",
        ]
        hparams = {k: v for k, v in arg_values.items() if k not in exclude_args}

        if args:
            # Save only specified arguments
            hparams = {arg: hparams[arg] for arg in args if arg in hparams}
        else:
            # Save all arguments except 'self'
            pass

        # Store in self.hparams
        self.hparams = Namespace(**hparams)

    def update_env(self, env: Env):
        self._env = env

    def log_scalar(self, name: str, value: float, episode: int):
        for logger in self._logger:
            logger.log_scalar(name, value, episode + self.episode_offset)

    def log_dict(self, d: Dict[str, float], episode: int, prefix: int):
        for logger in self._logger:
            logger.log_dict(d, episode + self.episode_offset, prefix)

    def save_metrics(self):
        for logger in self._logger:
            logger.save()

    def evaluate(self, episode_idx: int):
        """evaluate agent on eval environments

        Args:
            episode_idx (int): when this evaluate was deployed
        """
        if self._eval_envs is None:
            return

        # use the argmax agent to evaluate it
        agent = self.get_agent(deterministic=True)
        for eval_env in self._eval_envs:
            if eval_env is None:
                continue

            metrics = eval_env.eval_agent(agent)
            for k, v in metrics.items():
                self.log_scalar(k, v, episode_idx)

    def mid_training_hooks(self, episode_idx: int):
        if (
            self._save_interval is not None
            and (episode_idx + 1) % self._save_interval == 0
        ):
            # save model
            self.save_checkpoint(episode_idx + 1)

        if (
            self._eval_check_interval is not None
            and (episode_idx + 1) % self._eval_check_interval == 0
        ):
            self.evaluate(episode_idx + 1)

        self.save_metrics()

    def post_training_hooK(self, n_episodes: int):
        # store metrics in a csv file
        self.save_metrics()
        self.save_checkpoint(n_episodes)
        self._env.close()

    def set_episode_offset(self, offset: int):
        """sets the episode offset for logging. Particularly interesting if you start the train function multiple time on the same class to refine the agent further

        Args:
            offset (int): where the training should continue for logging
        """
        self.episode_offset = offset

    def get_basic_save_args(self, episode_idx: int) -> Dict[str, Any]:
        content = {
            "epoch": episode_idx,
            "hparams": vars(self.hparams),
            "state_dim": self._state_dim,  # assert observation space is a box
            "action_space": type(self._env.action_space).__name__,
        }

        if isinstance(self._env.action_space, Box):
            content = {
                **content,
                "action_dim": self._action_dim,
                "action_scale": self._action_scale.numpy(),
                "action_bias": self._action_bias.numpy(),
            }
        elif isinstance(self._env.action_space, Discrete):
            content = {**content, "n_actions": self._env.action_space.n}
        elif isinstance(self._env.action_space, MultiDiscrete):
            content = {**content, "nvec": self._env.action_space.nvec}
        return content

    @abstractmethod
    def save_checkpoint(self, episode_idx: int, path: Path | str = None):
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, checkpoint, env: Env = None) -> "_RLAlgorithm":
        checkpoint_content = torch.load(checkpoint, weights_only=False)
        if env is None:
            env = build_placeholder_env(checkpoint_content)

        algorithm: _RLAlgorithm = cls(env=env, **checkpoint_content["hparams"])
        algorithm.load_checkpoint(checkpoint)

        return algorithm

    def get_name(self) -> str:
        return type(self).__name__

    def to(self, device: torch.device) -> "_RLAlgorithm":
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device

        # port nn.Modules to device
        for value in self.__dict__.values():
            if not isinstance(value, nn.Module):
                continue
            value.to(self._device)
        return self

    @abstractmethod
    def load_checkpoint(self, checkpoint: str | Path):
        raise NotImplementedError

    @abstractmethod
    def train(self, n_episodes: int, verbose: bool = False):
        raise NotImplementedError

    @abstractmethod
    def get_agent(self, deterministic: bool = True) -> _Agent:
        raise NotImplementedError
