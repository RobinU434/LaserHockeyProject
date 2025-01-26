from abc import ABC, abstractmethod
from argparse import Namespace
import inspect
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
        self._log_dir = Path(self._log_dir) if not isinstance(self._log_dir, Path) else self._log_dir

        self.hparams = Namespace()

    def save_hyperparmeters(self, *args):
        # Get the frame of the calling function (i.e., the __init__ method)
        frame = inspect.currentframe().f_back
        
        # Get the arguments of the calling function
        arg_values = frame.f_locals
        
        # pre filter
        exclude_args = ["self", "env", "logger", "eval_env", "kwargs", "args", "__class__"]
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

    @abstractmethod
    def save_checkpoint(self, episode_idx: int, path: Path | str = None):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, config: SACConfig
    ) -> "RLAlgorithm":
        raise NotADirectoryError

    @abstractmethod
    def train(self, n_episodes: int):
        raise NotImplementedError
