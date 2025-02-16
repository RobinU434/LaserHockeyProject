import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np

from project.algorithms.common.algorithm import _RLAlgorithm
from project.environment.evaluate_env import EvalOpponent
from project.environment.hockey_env.hockey.hockey_env import Mode
from project.environment.single_player_env import SinglePlayerHockeyEnv


class _CheckpointSampler(ABC):
    def __init__(
        self,
        checkpoint_dir: Path,
        sample_interval: int = 100,
        name_pattern: str = r"checkpoint_(?P<episode>.+)\.pt",
    ):
        """_summary_

        Args:
            checkpoint_dir (Path): directory with checkpoints. Assert checkpoint naming scheme: "checkpoint_{epoch_idx}.pt"
        """
        super().__init__()
        self.checkpoint_dir = (
            Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir
        )
        self.sample_interval = sample_interval
        self.name_pattern = name_pattern
        self._current_episode = 0

    def _get_episodes(self) -> Tuple[List[int], List[Path]]:
        """get episode when the checkpoint was saved based  on the given naming pattern

        Returns:
            Tuple[List[int], List[Path]]: List of episode indices and list of checkpoint paths.
                indices and checkpoint are sorted in descending order
        """
        indices = []

        glob_pattern = re.sub(r"\(.*?\)\\", "*", self.name_pattern)
        checkpoints = list(self.checkpoint_dir.glob(glob_pattern))
        for path in checkpoints:
            file_name = path.name
            match = re.match(self.name_pattern, file_name)
            if match:
                indices.append(
                    int(match.group("episode"))
                )  # Extract the 'episode' value
            else:
                msg = f"Not able to extract episode from {file_name} with patter: {self.name_pattern}"
                logging.error(msg)
                indices.append(None)
        # sort indices
        sort_idx = np.argsort(indices)[::-1]
        indices = np.array(indices)[sort_idx].tolist()
        checkpoints = np.array(checkpoints)[sort_idx].tolist()
        return indices, checkpoints

    @abstractmethod
    def sample_checkpoint(self) -> Path:
        raise NotImplementedError

    def step(self):
        """increase step counter"""
        self._current_episode += 1

    def get_new_checkpoint(self) -> None | Path:
        """if it is time to get a new checkpoint the method will return a path to the new checkpoint.
        Otherwise it will return None

        Returns:
            None | Path: path to new checkpoint or nothing
        """
        if self._current_episode >= self.sample_interval:
            self._current_episode = 0  # reset counter
            # sample new checkpoint
            return self.sample_checkpoint()
        return None


class ExponentialSampler(_CheckpointSampler):
    """samples checkpoint name based on a exponential distribution. Older checkpoints will be sampled less."""

    def __init__(
        self,
        checkpoint_dir,
        lmbd: float = 0.2,
        sample_interval=100,
        name_pattern=r"checkpoint_(?P<episode>.+)\.pt",
    ):
        super().__init__(checkpoint_dir, sample_interval, name_pattern)
        self.lmbd = lmbd

    def sample_checkpoint(self):
        indices, checkpoints = self._get_episodes()
        x = np.array(indices, dtype=float)
        x -= x.min()
        x /= x.max()
        n_checkpoints = len(x)
        p = np.exp(-self.lmbd * x)
        p /= p.sum()

        sample_idx = np.random.choice(n_checkpoints, p=p)
        return checkpoints[sample_idx]


class WarmupSchedule:
    def __init__(
        self,
        n_episodes_weak: int = 0,
        n_episodes_strong: int = 0,
        keep_mode: bool = True,
        verbose: bool = False,
    ):
        """train first on weak opponent then on strong

        Args:
            n_episodes_weak (int, optional): _description_. Defaults to 0.
            n_episodes_strong (int, optional): _description_. Defaults to 0.
            keep_mode (bool, optional): Defaults to True
            verbose (bool, optional): Defaults to False
        """
        mode = Mode.NORMAL
        self.episode_counter = 0
        self.n_episodes_weak = n_episodes_weak
        self.n_episodes_strong = n_episodes_strong

        weak_opponent = EvalOpponent(weak=True, keep_mode=keep_mode, verbose=verbose)
        strong_opponent = EvalOpponent(weak=False, keep_mode=keep_mode, verbose=verbose)
        strong_opponent = EvalOpponent(weak=False, keep_mode=keep_mode, verbose=verbose)
        self.strong_env = SinglePlayerHockeyEnv(
            opponent=strong_opponent, keep_mode=keep_mode, mode=mode, verbose=verbose
        )

        weak_opponent = EvalOpponent(weak=True, keep_mode=keep_mode, verbose=verbose)
        self.weak_env = SinglePlayerHockeyEnv(
            opponent=weak_opponent, keep_mode=keep_mode, mode=mode, verbose=verbose
        )

    def step(self):
        self.episode_counter += 1

    def get_opponent(self) -> None | EvalOpponent:
        if self.is_terminated():
            return None

        if self.episode_counter <= self.n_episodes_weak:
            return self.weak_env
        elif (
            self.episode_counter > self.n_episodes_weak
            and self.episode_counter <= self.n_episodes_strong
        ):
            return self.strong_env
        else:
            logging.error("Unexpected episode counter value in WarmupSchedule")
            return None

    def is_terminated(self) -> bool:
        return self.episode_counter >= self.total_warmup_episodes()

    def total_warmup_episodes(self) -> int:
        return self.n_episodes_strong + self.n_episodes_weak


class SelfPlayTrainer:
    def __init__(
        self,
        env: SinglePlayerHockeyEnv,
        rl_algorithm: _RLAlgorithm,
        checkpoint_schedule: _CheckpointSampler = None,
        warmup_schedule: WarmupSchedule = None,
        verbose: bool = True,
    ):
        self.env = env
        self.rl_algorithm = rl_algorithm
        self.checkpoint_schedule = checkpoint_schedule
        self.warmup_schedule = warmup_schedule
        self.verbose = verbose

    def do_warmup(self, total_episode_budge: int):
        if self.warmup_schedule is None:
            return

        if total_episode_budge < self.warmup_schedule.total_warmup_episodes():
            logging.warning("Episode budget is not enough to complete warmup")

        # weak training
        self.rl_algorithm.update_env(self.warmup_schedule.weak_env)
        self.rl_algorithm.train(
            self.warmup_schedule.n_episodes_weak, verbose=self.verbose
        )
        self.rl_algorithm.episode_offset += self.warmup_schedule.n_episodes_weak
        # strong training
        self.rl_algorithm.update_env(self.warmup_schedule.strong_env)
        self.rl_algorithm.train(
            self.warmup_schedule.n_episodes_strong, verbose=self.verbose
        )
        self.rl_algorithm.episode_offset += self.warmup_schedule.n_episodes_strong

    def train(self, n_episodes: int):
        self.do_warmup(n_episodes)
        self.rl_algorithm.save_checkpoint(self.warmup_schedule.total_warmup_episodes())
        remaining_episode_budged = (
            n_episodes - self.warmup_schedule.total_warmup_episodes()
        )

        budgets = [self.checkpoint_schedule.sample_interval] * (
            remaining_episode_budged // self.checkpoint_schedule.sample_interval
        )
        rest_budged = remaining_episode_budged - np.sum(budgets, dtype=int)
        if rest_budged > 0:
            budgets.append(rest_budged)

        for self_play_budget in budgets:
            # load past checkpoint as current opponent
            checkpoint = self.checkpoint_schedule.sample_checkpoint()
            self_play_agent = (
                type(self.rl_algorithm)
                .from_checkpoint(checkpoint)
                .get_agent(deterministic=False)
            )
            # update env
            self.env.opponent = self_play_agent
            self.rl_algorithm.update_env(self.env)
            # start training
            self.rl_algorithm.train(self_play_budget, verbose=self.verbose)
            self.rl_algorithm.episode_offset += self_play_budget
