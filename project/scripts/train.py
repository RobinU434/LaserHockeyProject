from pathlib import Path
import gymnasium
import hydra
from gymnasium.spaces import Box, Discrete
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.logger import configure
import logging

from project.algorithms.common.logger import CSVLogger, TensorBoardLogger
from project.algorithms.env_wrapper import (
    DiscreteActionWrapper,
    SymLogWrapper,
    TanhWrapper,
)
from project.algorithms.sac.sac import SAC
from project.algorithms.trainer import (
    ExponentialSampler,
    SelfPlayTrainer,
    WarmupSchedule,
)
from project.environment.evaluate_env import EvalHockeySuite
from project.environment.hockey_env.hockey.hockey_env import HockeyEnv
from project.environment.single_player_env import SinglePlayerHockeyEnv
from project.utils.configs.train_sac_config import Config as SACConfig
from gymnasium.wrappers import NormalizeReward

# from project.environment.hockey_env.hdf5_replay_buffer import HDF5ReplayBuffer


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


def train_sac(config: DictConfig, force: bool = False, device: str = "cpu"):
    config: SACConfig = SACConfig.from_dict_config(config)
    # build envs (train, eval env)
    train_env = SinglePlayerHockeyEnv(**config.SelfPlay.Env.to_container())
    eval_env = EvalHockeySuite(**config.SelfPlay.Env.to_container())

    
    # ititialize HDF5ReplayBuffer (save interactions)
    # hdf5_replay_buffer = HDF5ReplayBuffer("interactions.h5", max_size=100000)

    # build algorithm
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = [TensorBoardLogger(log_dir), CSVLogger(log_dir)]
    sac = SAC(
        train_env,
        logger=logger,
        eval_env=eval_env,
        eval_check_interval=config.eval_check_interval,
        save_interval=config.save_interval,
        log_dir=log_dir,
        **config.SAC.to_container(),
    )
    sac.to(device)

    # build trainer
    checkpoint_schedule = ExponentialSampler(
        log_dir, sample_interval=config.SelfPlay.self_play_period
    )
    warmup_schedule = WarmupSchedule(**config.SelfPlay.WarmupSchedule.to_container())
    trainer = SelfPlayTrainer(
        env=train_env,
        rl_algorithm=sac,
        checkpoint_schedule=checkpoint_schedule,
        warmup_schedule=warmup_schedule,
    )

    print(sac)

    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", "yes"]):
            print("Abort training")
            return

    # train loop, saves interactions in hdf5
    """for episode in range(config.episode_budget):
        state = train_env.reset()
        done = False
        while not done:
            action = sac.select_action(state)  
            next_state, reward, done, _ = train_env.step(action)

            # save interactions
            hdf5_replay_buffer.add_interaction(state, action, reward, next_state, done)

            # prepare next state for interactions
            state = next_state

        # if there is a predefined checkpoint for saving, save the buffer
        if (episode + 1) % config.save_interval == 0:
            print(f"Saving HDF5 buffer after {episode + 1} episodes.")
            hdf5_replay_buffer.close()
    """
    trainer.train(config.episode_budget)
    # train algorithm


def train_sac_gym_env(
    config: DictConfig,
    force: bool,
    gym_env: str,
    max_steps: int = 200,
    device: str = "cpu",
):
    config: SACConfig = SACConfig.from_dict_config(config)
    # build envs (train, eval env)
    train_env = gymnasium.make(gym_env, max_episode_steps=max_steps)
    if gym_env == "LunarLander-v3":
        train_env = gymnasium.make(
            "LunarLander-v3", continuous=True, max_episode_steps=max_steps
        )
    # train_env = TanhWrapper(train_env, 1000)
    train_env = SymLogWrapper(train_env)

    print("Train on environment: ", gym_env)
    action_space: Box = train_env.action_space
    config.SAC.action_scale = action_space.high - action_space.low
    config.SAC.action_bias = (action_space.high + action_space.low) / 2.0

    # build algorithm
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    subfolder_name = gym_env.split("-")[0]
    log_dir = log_dir.parent / subfolder_name / log_dir.name
    logger = [TensorBoardLogger(log_dir), CSVLogger(log_dir)]
    sac = SAC(
        train_env,
        logger=logger,
        save_interval=config.save_interval,
        log_dir=log_dir,
        **config.SAC.to_container(),
    )
    sac = sac.to(device)
    print(sac)

    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", "yes"]):
            print("Abort training")
            return

    # train algorithm
    sac.train(config.episode_budget)


def train_dyna_gym_env(
    config: DictConfig,
    force: bool,
    gym_env: str,
    max_steps: int = 200,
    n_actions: int = 10,
    device: str = "cpu",
):
    config: SACConfig = SACConfig.from_dict_config(config)
    # build envs (train, eval env)
    train_env = gymnasium.make(gym_env, max_episode_steps=max_steps)
    if isinstance(train_env, Box):
        assert train_env.action_space.shape == (
            1,
        ), "For Dyna you need a one dimension in the action space"
        train_env = DiscreteActionWrapper(train_env, n_actions)
    else:
        assert isinstance(
            train_env.action_space, Discrete
        ), "asserted Discrete action space in  given environment"
        logging.warning(
            f"Argument n_actions will be ignored. Instead use n_actions from given actions space: n={train_env.action_space.n}"
        )

    train_env = SymLogWrapper(train_env)

    print("Train on environment: ", gym_env)
    action_space: Box = train_env.action_space
    config.SAC.action_scale = action_space.high - action_space.low
    config.SAC.action_bias = (action_space.high + action_space.low) / 2.0

    # build algorithm
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    subfolder_name = gym_env.split("-")[0]
    log_dir = log_dir.parent / subfolder_name / log_dir.name
    logger = [TensorBoardLogger(log_dir), CSVLogger(log_dir)]
    sac = SAC(
        train_env,
        logger=logger,
        save_interval=config.save_interval,
        log_dir=log_dir,
        **config.SAC.to_container(),
    )
    sac = sac.to(device)
    print(sac)

    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", "yes"]):
            print("Abort training")
            return

    # train algorithm
    sac.train(config.episode_budget)
