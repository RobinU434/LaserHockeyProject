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

    @add_hydra("config", None, config_path="config", config_name="train_sb_sac.yaml")
    def train_sb3_sac_hockey(
        self, config: DictConfig, force: bool = False, device: str = "cpu"
    ):
        from project.scripts.train import train_sb3_sac

        train_sb3_sac(config, force, device)

    @add_hydra("config", None, config_path="config", config_name="train_sb_sac.yaml")
    def train_sb3_sac_sp(
        self, config: DictConfig, force: bool = False, device: str = "cpu"
    ):
        from project.scripts.train import train_sb3_sac_sp

        train_sb3_sac_sp(config, force, device)

    @add_hydra("config", None, config_path="config", config_name="train_sb_sac.yaml")
    def train_sb3_sac_gym(
        self, config: DictConfig, gym_env: str, force: bool = False, device: str = "cpu"
    ):
        from project.scripts.train import train_sb3_sac_gym

        train_sb3_sac_gym(config, gym_env, force, device)

    @add_hydra("config", None, config_path="config", config_name="train_sb_sac.yaml")
    def train_sb3_er_sac_gym(
        self, config: DictConfig, gym_env: str, force: bool = False, device: str = "cpu"
    ):
        from project.scripts.train import train_sb3_er_sac_gym

        train_sb3_er_sac_gym(config, gym_env, force, device)

    def upload_sb3_sac(
        self, checkpoint: str, server_url: str, server_port: int, token: str
    ):
        """upload sb3 sac agent to competition server

        Args:
            checkpoint (str): path to SB3 sac checkpoint
            server_url (str): URL of the server.
            server_port (int): Port of the server.
            token (str): Your access token.
        """
        from project.scripts.upload import upload_sb_sac_agent

        upload_sb_sac_agent(checkpoint, server_url, server_port, token)

    @add_hydra("config", None, config_path="config", config_name="train_sb_sac.yaml")
    def train_sb3_er_sac_hockey(
        self, config: DictConfig, force: bool = False, device: str = "cpu"
    ):
        from project.scripts.train import train_sb3_er_sac

        train_sb3_er_sac(config, force, device)

    @add_hydra("config", None, config_path="config", config_name="train_sac.yaml")
    def train_sac_hockey(
        self, config: DictConfig, force: bool = False, device: str = "cpu"
    ):
        from project.scripts.train import train_sac

        train_sac(config, force, device)

    def render_sac_hockey(
        checkpoint: str, deterministic: bool = False, strong_opponent: bool = False
    ):
        from project.scripts.render import render_sac_hockey

        render_sac_hockey(checkpoint, deterministic, strong_opponent)

    @add_hydra("config", None, config_path="config", config_name="train_sac.yaml")
    def train_sac_gym(
        self,
        config: DictConfig,
        gym_env: str,
        force: bool = False,
        max_steps: int = 200,
        device: str = "cpu",
        quiet: bool = False,
    ):
        from project.scripts.train import train_sac_gym_env

        train_sac_gym_env(config, force, gym_env, max_steps, device, quiet)

    def eval_sac(self, checkpoint: str, n_games: int = 10, deterministic: bool = False):
        from project.scripts.evaluate import evaluate_sac

        evaluate_sac(checkpoint, n_games, deterministic)

    def render_sac_gym(
        self,
        checkpoint: str,
        gym_env: str,
        deterministic: bool = False,
        max_steps: int = 200,
    ):
        from project.scripts.render import render_sac

        render_sac(checkpoint, gym_env, deterministic, max_steps)

    @add_hydra("config", None, config_path="config", config_name="train_dyna.yaml")
    def train_dyna_hockey(
        self,
        config: DictConfig,
        force: bool = False,
        device: str = "cpu",
        quiet: bool = False,
    ):
        from project.scripts.train import train_dyna_hockey

        train_dyna_hockey(config, force, device, quiet)

    @add_hydra("config", None, config_path="config", config_name="train_dyna.yaml")
    def train_dyna_gym(
        self,
        config: DictConfig,
        gym_env: str,
        n_actions: int = 10,
        force: bool = False,
        quiet: bool = False,
        max_steps: int = 200,
        device: str = "cpu",
    ):
        from project.scripts.train import train_dyna_gym_env

        train_dyna_gym_env(config, force, gym_env, max_steps, n_actions, device, quiet)

    def render_dyna_gym(
        self,
        checkpoint: str,
        gym_env: str,
        deterministic: bool = False,
        max_steps: int = 200,
    ):
        from project.scripts.render import render_dyna

        render_dyna(checkpoint, gym_env, deterministic, max_steps)

    @add_hydra("config", None, config_path="config", config_name="train_md_dyna.yaml")
    def train_md_dyna_gym(
        self,
        config: DictConfig,
        gym_env: str,
        n_actions: int = 10,
        force: bool = False,
        quiet: bool = False,
        max_steps: int = 200,
        device: str = "cpu",
    ):
        from project.scripts.train import train_md_dyna_gym_env

        train_md_dyna_gym_env(
            config, force, gym_env, max_steps, n_actions, device, quiet
        )

    @add_hydra("config", None, config_path="config", config_name="train_er_dyna.yaml")
    def train_er_dyna_gym(
        self,
        config: DictConfig,
        gym_env: str,
        n_actions: int = 10,
        force: bool = False,
        quiet: bool = False,
        max_steps: int = 200,
        device: str = "cpu",
    ):
        from project.scripts.train import train_er_dyna_gym_env

        train_er_dyna_gym_env(
            config, force, gym_env, max_steps, n_actions, device, quiet
        )

    @add_hydra("config", None, config_path="config", config_name="train_er_dyna.yaml")
    def train_er_dyna_hockey(
        self,
        config: DictConfig,
        force: bool = False,
        quiet: bool = False,
        device: str = "cpu",
    ):
        from project.scripts.train import train_er_dyna_hockey

        train_er_dyna_hockey(config, force, device, quiet)

    def upload_dyna(
        self, checkpoint: str, server_url: str, server_port: int, token: str
    ):
        """upload sb3 sac agent to competition server

        Args:
            checkpoint (str): path to SB3 sac checkpoint
            server_url (str): URL of the server.
            server_port (int): Port of the server.
            token (str): Your access token.
        """
        from project.scripts.upload import upload_dyna_agent

        upload_dyna_agent(checkpoint, server_url, server_port, token)
