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
    def train_sac(self, config: DictConfig, force: bool = False):
        from project.scripts.train import train_sac
        train_sac(config, force)

    @add_hydra("config  ")
    def train_pendulum(self, config: DictConfig):
        pass