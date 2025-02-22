from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
from gymnasium import Env
import gymnasium
from tqdm import tqdm

from project.algorithms.common.agent import _Agent
from project.environment.hockey_env.hockey.hockey_env import Mode
from project.environment.opponent import EvalOpponent
from project.environment.single_player_env import SinglePlayerHockeyEnv


class _EvalEnv(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval_agent(self, agent: _Agent) -> Dict[str, float]:
        """do agent evaluation and return metrics / statistics of this evaluation

        Args:
            agent (_Agent): agent to evaluate

        Returns:
            Dict[str, float]: dict with names and statistics
        """
        raise NotADirectoryError


class EvalGymSuite(_EvalEnv):
    def __init__(self, env: Env, n_episodes: int = 1):
        super().__init__()

        self.env = env
        self.n_episodes = n_episodes

    @classmethod
    def make(cls, env_name: str, n_games: int = 1, *args, **kwargs) -> "EvalGymSuite":
        env = gymnasium.make(env_name, *args, **kwargs)
        obj = cls(env, n_games)
        return obj

    def eval_agent(self, agent):
        results = []
        iterator = range(self.n_games)
        if self.verbose:
            iterator = tqdm(iterator, desc="Evaluate", unit="game")

        for _ in range(self.n_games):
            results.append(self._collect_rollout(agent))

        results = pd.DataFrame(results)
        mean_results = results.mean()
        mean_results = mean_results.to_dict()
        # add prefix
        mean_results = {k: v for k, v in mean_results.items()}
        return mean_results

    def collect_rollout(self, agent: _Agent) -> Dict[str, float]:
        score = 0
        step_counter = 0

        done, truncated = False, False
        state, _ = self.env.reset()
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            step_counter += 1
            score += reward

            state = next_state

        mean_score = score / step_counter
        mean_length = step_counter

        results = {
            "mean_score": mean_score,
            "mean_length": mean_length,
        }

        return results


class EvalHockeySuite(_EvalEnv):
    def __init__(
        self,
        keep_mode=True,
        mode=Mode.NORMAL,
        verbose=False,
        n_games: int = 10,
    ):
        super().__init__()
        self.n_games = n_games  # how many games you want to play per opponent
        self.verbose = verbose

        strong_opponent = EvalOpponent(weak=False, keep_mode=keep_mode, verbose=verbose)
        self.strong_env = SinglePlayerHockeyEnv(
            opponent=strong_opponent, keep_mode=keep_mode, mode=mode, verbose=verbose
        )

        weak_opponent = EvalOpponent(weak=True, keep_mode=keep_mode, verbose=verbose)
        self.weak_env = SinglePlayerHockeyEnv(
            opponent=weak_opponent, keep_mode=keep_mode, mode=mode, verbose=verbose
        )

    def eval_agent(self, player: _Agent, verbose: bool = False) -> Dict[str, float]:
        """evaluate player on the weak and strong opponent

        Args:
            player (Agent): player to evaluate
            verbose (bool, optional): show progress bar

        Returns:
            Dict[str, float]: dict with metrics to log further on. Contains:
                - eval_strong/win_rate
                - eval_strong/tie_rate
                - eval_strong/loose_rate
                - eval_strong/mean_score
                - eval_strong/mean_length
                - eval_weak/win_rate
                - eval_weak/tie_rate
                - eval_weak/loose_rate
                - eval_weak/mean_score
                - eval_weak/mean_length
        """
        strong_results = self._eval_on_opponent(
            player, self.strong_env, prefix="eval_strong/", verbose=verbose
        )
        weak_results = self._eval_on_opponent(
            player, self.weak_env, prefix="eval_weak/", verbose=verbose
        )
        results = {**strong_results, **weak_results}
        return results

    def _eval_on_opponent(
        self, player: _Agent, env: Env, prefix: str = "", verbose: bool = False
    ) -> Dict[str, float]:
        """evaluate player on environment

        Args:
            player (Agent): player to evaluate
            env (Env): env to evaluate on
            prefix (str, optional): prefix for log tag / name. Defaults to "".
            verbose (bool, optional): show progress bar.

        Returns:
            Dict[str, float]: log metrics:
            - (prefix)win_rate
            - (prefix)tie_rate
            - (prefix)loose_rate
            - (prefix)mean_score
            - (prefix)mean_length
        """
        results = []
        iterator = range(self.n_games)
        if verbose:
            iterator = tqdm(iterator, desc="Evaluate", unit="game")

        for _ in iterator:
            results.append(self._collect_rollout(player, env))

        # aggregate results
        results = pd.DataFrame(results)
        mean_results = results.mean()
        mean_results = mean_results.to_dict()
        # add prefix
        mean_results = {f"{prefix}{k}": v for k, v in mean_results.items()}
        return mean_results

    def _collect_rollout(self, player: _Agent, env: Env) -> Dict[str, float]:
        """do one game

        Args:
            player (Agent): _description_
            env (Env): _description_

        Returns:
            Dict[str, float]: metrics like
                - win_rate
                - tie_rate
                - loose_rate
                - mean_score
                - mean_length
        """

        score = 0.0
        last_score = 0
        step_counter = 0

        done = False
        truncated = False

        state, _ = env.reset()
        while not (done or truncated):
            action = player.act(state)
            state, reward, done, truncated, _ = env.step(action)

            score += reward
            last_score = reward
            step_counter += 1

        if last_score > 0:
            win_rate = 1
            tie_rate = 0
            loose_rate = 0
        elif last_score == 0:
            win_rate = 0
            tie_rate = 1
            loose_rate = 0
        else:
            win_rate = 0
            tie_rate = 0
            loose_rate = 1

        mean_score = score / step_counter
        mean_length = step_counter

        results = {
            "win_rate": win_rate,
            "tie_rate": tie_rate,
            "loose_rate": loose_rate,
            "mean_score": mean_score,
            "mean_length": mean_length,
        }

        return results
