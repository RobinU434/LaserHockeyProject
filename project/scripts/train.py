from project.environment.hockey_env.hockey.hockey_env import HockeyEnv
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3

from stable_baselines3.common.logger import configure
from project.utils.configs.train_sac_config import Config as SACConfig

def train_dreamer():
    pass


def train_sb3_sac():
    env = HockeyEnv(mode="NORMAL")

    tmp_path = "/tmp/sb3_log/"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # model = SAC("MlpPolicy", env, verbose=1)
    model = SAC("MlpPolicy", env, verbose=1)
    model.set_logger(new_logger)

    model.learn(100)


def train_sac(config: SACConfig):
    
