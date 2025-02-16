from omegaconf import DictConfig
from pyargwriter.decorator import add_hydra


class Entrypoint:
    """Entrypoint to experimentation landscape"""

    def __init__(self):
        """init object instance"""
        pass

    def train_dreamer(self):
        """_summary_"""
        pass

    def train_sb3_sac(self):
        """_summary_"""
        from project.scripts.train import train_sb3_sac

        train_sb3_sac()

    @add_hydra("config", None, config_path="config", config_name="train_sac.yaml")
    def train_sac_hockey(
        self, config: DictConfig, force: bool = False, device: str = "cpu"
    ):
        from project.scripts.train import train_sac

        train_sac(config, force, device)

    @add_hydra("config", None, config_path="config", config_name="train_sac.yaml")
    def train_sac_gym(
        self,
        config: DictConfig,
        gym_env: str,
        force: bool = False,
        max_steps: int = 200,
        device: str = "cpu",
    ):
        from project.scripts.train import train_sac_gym_env

        train_sac_gym_env(config, force, gym_env, max_steps, device)

    def render_sac_gym(
        self,
        checkpoint: str,
        gym_env: str,
        deterministic: bool = False,
        max_steps: int = 200,
    ):
        from project.scripts.render import render_sac

        render_sac(checkpoint, gym_env, deterministic, max_steps)

    def eval_sac(self, checkpoint: str, n_games: int = 10, deterministic: bool = False):
        from project.scripts.evaluate import evaluate_sac

        evaluate_sac(checkpoint, n_games, deterministic)
