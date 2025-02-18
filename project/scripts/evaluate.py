from pathlib import Path

from project.algorithms.sac.sac import SAC
from project.environment.evaluate_env import EvalHockeySuite


def evaluate_sac(checkpoint: Path, n_games: int, deterministic: bool = False):
    sac = SAC.from_checkpoint(checkpoint)
    eval_env = EvalHockeySuite(n_games=n_games, verbose=True)
    print(f"evaluate {n_games} games")
    results = eval_env.eval_agent(sac.get_agent(deterministic))
    print(results)
