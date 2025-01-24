from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from gymnasium import Env
from project.algorithms.logger import _Logger
from project.utils.configs.train_sac_config import Config as SACConfig


class RLAlgorithm(ABC):
    def __init__(
        self,
        env: Env,
        logger: List[_Logger] = [],
        eval_env: List[Env] = None,
        eval_check_interval: int = None,
        save_interval: int = None,
        log_dir: Path = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self._env = env
        self._logger = logger
        self._eval_env = eval_env
        self._eval_check_interval = eval_check_interval
        self._save_interval = save_interval
        self._log_dir = (
            log_dir if log_dir is not None else Path().cwd().joinpath("results/")
        )

    def update_env(self, env: Env):
        self._env = env

    def log_scalar(self, name: str, value: float, step: int):
        for logger in self._logger:
            logger.log_scalar(name, value, step)

    def save_metrics(self):
        for logger in self._logger:
            logger.save()

    @abstractmethod
    def save_checkpoint(self, episode_idx: int, path: Path | str = None):
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, config: SACConfig
    ) -> "RLAlgorithm":
        raise NotADirectoryError

    @abstractmethod
    def train(self, n_episodes: int):
        raise NotImplementedError
