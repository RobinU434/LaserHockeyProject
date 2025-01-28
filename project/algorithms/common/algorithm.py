import inspect
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import List

import torch
from gymnasium import Env

from project.algorithms.common.agent import _Agent
from project.algorithms.common.logger import _Logger
from project.algorithms.utils import PlaceHolderEnv
from project.environment.evaluate_env import _EvalEnv


class _RLAlgorithm(ABC):
    def __init__(
        self,
        env: Env,
        logger: List[_Logger] = [],
        eval_env: List[_EvalEnv] | _EvalEnv = None,
        eval_check_interval: int = None,
        save_interval: int = None,
        log_dir: Path = None,
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

        self.hparams = Namespace()

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

    def log_scalar(self, name: str, value: float, step: int):
        for logger in self._logger:
            logger.log_scalar(name, value, step)

    def save_metrics(self):
        for logger in self._logger:
            logger.save()

    def evaluate(self, episode_idx: int):
        """evaluate agent on eval environments

        Args:
            episode_idx (int): when this evaluate was deployed
        """
        # use the argmax agent to evaluate it
        agent = self.get_agent(deterministic=True)
        for eval_env in self._eval_envs:
            metrics = eval_env.eval_player(agent)
            for k, v in metrics.items():
                self.log_scalar(k, v, episode_idx)

    @abstractmethod
    def save_checkpoint(self, episode_idx: int, path: Path | str = None):
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, checkpoint, env: Env = None) -> "_RLAlgorithm":
        checkpoint_content = torch.load(checkpoint, weights_only=True)
        if env is None:
            env = PlaceHolderEnv(
                checkpoint_content["state_dim"], checkpoint_content["action_dim"]
            )

        sac: _RLAlgorithm = cls(env=env, **checkpoint_content["hparams"])
        sac.load_checkpoint(checkpoint)

        return sac

    @abstractmethod
    def load_checkpoint(self, checkpoint: str | Path):
        raise NotImplementedError

    @abstractmethod
    def train(self, n_episodes: int):
        raise NotImplementedError

    @abstractmethod
    def get_agent(self, deterministic: bool = True) -> _Agent:
        raise NotImplementedError
