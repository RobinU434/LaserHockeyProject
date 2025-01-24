from typing import Dict
import numpy as np
import pandas as pd
from project.algorithms.agent import Agent
from project.environment.hockey_env.hockey.hockey_env import BasicOpponent, HockeyEnv, Mode
from gymnasium import Env, spaces

class SinglePlayerHockeyEnv(HockeyEnv):
    """Agent to train is always player 1

    Args:
        HockeyEnv (_type_): _description_
    """
    def __init__(
        self,
        opponent: object,  # the other agent
        keep_mode=True,
        mode=Mode.NORMAL,
        verbose=False,
    ):
        super().__init__(keep_mode, mode, verbose)
        self.opponent = opponent
        self.action_space = spaces.Box(-1, +1, (self.num_actions,), dtype=np.float32)


    def reset(self, opponent_starts: bool = None, mode = None, seed = None, options = None):
        """reset environment
        assume: opponent has always the second half of the action (player two)

        Args:
            opponent_starts (bool, optional): opponent has the puck first. Defaults to None.
            mode (_type_, optional): _description_. Defaults to None.
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[np.ndarray, dict]: state, info
        """
        if opponent_starts is not None:
            player_starts = not opponent_starts
        else:
            player_starts = opponent_starts         
        
        return super().reset(one_starting=player_starts, mode=mode, seed=seed, options=options)
    
    def step(self, action: np.ndarray):
        obs_agent_two = self.opponent.self.obs_agent_two()



class EvalOpponent(Agent):
    def __init__(self, weak: bool = True, keep_mode: bool = True, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.opponent = BasicOpponent(weak, keep_mode)

    def act(self, state):
        return self.opponent.act(state, self.verbose)


class EvalHockeEnv:
    def __init__(self, keep_mode=True, mode=Mode.NORMAL, verbose=False, n_games: int = 10,):
        super().__init__()
        self.n_games = n_games  # how many games you want to play per opponent

        strong_opponent = EvalOpponent(weak=False, keep_mode=keep_mode, verbose=verbose)
        self.strong_env = SinglePlayerHockeyEnv(opponent=strong_opponent, keep_mode=keep_mode, mode=mode, verbose=verbose)

        weak_opponent = EvalOpponent(weak=True, keep_mode=keep_mode, verbose=verbose)
        self.weak_env = SinglePlayerHockeyEnv(opponent=weak_opponent, keep_mode=keep_mode, mode=mode, verbose=verbose)

    def eval_player(self, player: Agent) -> Dict[str, float]:
        """evaluate player on the weak and strong opponent

        Args:
            player (Agent): player to evaluate

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
        strong_results = self._eval_on_opponent(player, self.strong_env, prefix="eval_strong/")
        weak_results = self._eval_on_opponent(player, self.weak_env, prefix="eval_weak/")
        results = {**strong_results, **weak_results}
        return results
    
    def _eval_on_opponent(self, player: Agent, env: Env, prefix: str = "") -> Dict[str, float]:
        """evaluate player on environment

        Args:
            player (Agent): player to evaluate
            env (Env): env to evaluate on
            prefix (str, optional): prefix for log tag / name. Defaults to "".

        Returns:
            Dict[str, float]: log metrics:
            - (prefix)win_rate
            - (prefix)tie_rate
            - (prefix)loose_rate
            - (prefix)mean_score
            - (prefix)mean_length
        """
        results = []
        for _ in self.n_games:
            results.append(self._collect_rollout(player, env))
        
        # aggregate results
        results: pd.DataFrame = pd.Dataframe(results)
        mean_results = results.mean()
        mean_results: dict = mean_results.to_dict()
        # add prefix
        mean_results = {f"{prefix}{k}": v for k, v in mean_results.items()}
        return mean_results

    def _collect_rollout(self, player: Agent, env: Env) -> Dict[str, float]:
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

