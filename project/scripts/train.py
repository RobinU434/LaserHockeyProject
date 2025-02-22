from pathlib import Path
import gymnasium
import hydra
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.logger import configure
from stable_baselines3.sac import SAC as SB_SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
import logging

from project.algorithms.common.logger import CSVLogger, TensorBoardLogger
from project.algorithms.dyna.dyna import DynaQ, MultiDiscreteDynaQ
from project.algorithms.dyna.experimental import ERDynaQ
from project.algorithms.env_wrapper import (
    AffineActionTransform,
    Box2DiscreteActionWrapper,
    Box2MultiDiscreteActionWrapper,
    MD2DiscreteActionWrapper,
    SymLogWrapper,
    TanhWrapper,
)
from project.algorithms.sb3_extensions.sac import ER_SAC, ERReplayBuffer
from project.algorithms.sac.sac import SAC
from project.algorithms.trainer import (
    ExponentialSampler,
    SelfPlayTrainer,
    WarmupSchedule,
)
from project.environment.evaluate_env import EvalGymSuite, EvalHockeySuite
from project.environment.hockey_env.hockey.hockey_env import BasicOpponent, HockeyEnv
from project.environment.single_player_env import SinglePlayerHockeyEnv
from project.utils.configs.train_sac_config import Config as SACConfig
from project.utils.configs.train_sb_sac_config import Config as SBSACConfig
from project.utils.configs.train_dyna_config import Config as DynaConfig
from project.utils.configs.train_er_dyna_config import Config as ERDynaConfig
from project.utils.configs.train_md_dyna_config import Config as MDDynaConfig
from gymnasium.wrappers import NormalizeReward

# from project.environment.hockey_env.hdf5_replay_buffer import HDF5ReplayBuffer


def train_dreamer():
    pass


def train_sb3_sac(config: DictConfig, force: bool = False, device: str = "cpu"):
    config: SBSACConfig = SBSACConfig.from_dict_config(config)
    opponent = BasicOpponent(weak=False)
    train_env = SinglePlayerHockeyEnv(opponent, **config.SelfPlay.Env.to_container())
    train_env = AffineActionTransform(
        train_env, np.array([1, 1, 1, 0.5]), np.array([0, 0, 0, 0.5])
    )
    # build
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Set new logger
    sac = SB_SAC(
        "MlpPolicy",
        env=train_env,
        tensorboard_log=log_dir,
        device=device,
        verbose=1,
        **config.SAC.to_container(),
    )
    # set up logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    sac.set_logger(new_logger)

    print(sac)
    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", " yes"]):
            print("Abort training")
            return

    callback = CheckpointCallback(config.save_interval, log_dir, "sac_model")
    sac.learn(config.episode_budget, callback=callback)


def train_sb3_er_sac(config: DictConfig, force: bool = False, device: str = "cpu"):
    config: SBSACConfig = SBSACConfig.from_dict_config(config)
    opponent = BasicOpponent(weak=False)
    train_env = SinglePlayerHockeyEnv(opponent, **config.SelfPlay.Env.to_container())
    train_env = AffineActionTransform(
        train_env, np.array([1, 1, 1, 0.5]), np.array([0, 0, 0, 0.5])
    )
    # build
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Set new logger
    sac = ER_SAC(
        "MlpPolicy",
        env=train_env,
        tensorboard_log=log_dir,
        device=device,
        verbose=1,
        replay_buffer_class=ERReplayBuffer,
        **config.SAC.to_container(),
    )
    # set up logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    sac.set_logger(new_logger)

    print(sac)
    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", " yes"]):
            print("Abort training")
            return

    callback = CheckpointCallback(config.save_interval, log_dir, "sac_model")
    sac.learn(config.episode_budget, callback=callback)


def train_sac(config: DictConfig, force: bool = False, device: str = "cpu"):
    config: SACConfig = SACConfig.from_dict_config(config)
    # build envs (train, eval env)
    train_env = SinglePlayerHockeyEnv(**config.SelfPlay.Env.to_container())
    eval_env = EvalHockeySuite(**config.SelfPlay.Env.to_container())

    train_env = AffineActionTransform(
        train_env, np.array([1, 1, 1, 0.5]), np.array([0, 0, 0, 0.5])
    )

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

    trainer.train(config.episode_budget)
    # train algorithm


def train_sac_gym_env(
    config: DictConfig,
    force: bool,
    gym_env: str,
    max_steps: int = 200,
    device: str = "cpu",
    quiet: bool = False,
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
    eval_env = EvalGymSuite(train_env, n_episodes=10)

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
        eval_env=[eval_env],
        logger=logger,
        log_dir=log_dir,
        **config.SAC.to_container(),
    )
    sac = sac.to(device)
    if not quiet:
        print("Train on environment: ", gym_env)
        print(config)
        print(sac)

    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", "yes"]):
            print("Abort training")
            return

    # train algorithm
    sac.train(config.episode_budget, verbose=not quiet)


def train_dyna_gym_env(
    config: DictConfig,
    force: bool,
    gym_env: str,
    max_steps: int = 200,
    n_actions: int = 10,
    device: str = "cpu",
    quiet: bool = False,
):
    config: DynaConfig = DynaConfig.from_dict_config(config)
    # build envs (train, eval env)
    train_env = gymnasium.make(gym_env, max_episode_steps=max_steps)
    if isinstance(train_env.action_space, Box):
        assert train_env.action_space.shape == (
            1,
        ), "For Dyna you need a one dimension in the action space"
        train_env = Box2DiscreteActionWrapper(train_env, n_actions)
    else:
        assert isinstance(
            train_env.action_space, Discrete
        ), "asserted Discrete action space in  given environment"
        logging.warning(
            f"Argument n_actions will be ignored. Instead use n_actions from given actions space: n={train_env.action_space.n}"
        )

    eval_env = EvalGymSuite(train_env, n_episodes=10)
    train_env = SymLogWrapper(train_env)

    # build algorithm
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    subfolder_name = gym_env.split("-")[0]
    log_dir = log_dir.parent / subfolder_name / log_dir.name
    logger = [TensorBoardLogger(log_dir), CSVLogger(log_dir)]

    dyna = DynaQ(
        env=train_env,
        eval_env=[eval_env],
        logger=logger,
        log_dir=log_dir,
        device=device,
        **config.Dyna.to_container(),
    )
    dyna.to(device)

    if not quiet:
        print("Train on environment: ", gym_env)
        print(config)
        print(dyna)

    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", "yes"]):
            print("Abort training")
            return

    # train algorithm
    dyna.train(config.episode_budget, verbose=not quiet)


def train_er_dyna_gym_env(
    config: DictConfig,
    force: bool,
    gym_env: str,
    max_steps: int = 200,
    n_actions: int = 10,
    device: str = "cpu",
    quiet: bool = False,
):
    """train the experimental version of Dyna

    Args:
        config (DictConfig): _description_
        force (bool): _description_
        gym_env (str): _description_
        max_steps (int, optional): _description_. Defaults to 200.
        n_actions (int, optional): _description_. Defaults to 10.
        device (str, optional): _description_. Defaults to "cpu".
        quiet (bool, optional): _description_. Defaults to False.
    """

    config: ERDynaConfig = ERDynaConfig.from_dict_config(config)
    # build envs (train, eval env)
    env_kwarg = {}
    if "LunarLander" in gym_env:
        env_kwarg["continuous"] = True
    train_env = gymnasium.make(gym_env, max_episode_steps=max_steps, **env_kwarg)
    if (
        isinstance(train_env.action_space, Box)
        and train_env.action_space.shape == (1,)
        and train_env.action_space[0] == 1
    ):
        train_env = Box2DiscreteActionWrapper(train_env, n_actions)
    if (
        isinstance(train_env.action_space, Box)
        and len(train_env.action_space.shape) == 1
        and train_env.action_space.shape[0] > 1
    ):
        nvec = np.ones(train_env.action_space.shape) * n_actions
        train_env = Box2MultiDiscreteActionWrapper(train_env, nvec)
        train_env = MD2DiscreteActionWrapper(train_env)
    elif isinstance(train_env.action_space, Discrete):
        logging.warning(
            f"Argument n_actions will be ignored. Instead use n_actions from given actions space: n={train_env.action_space.n}"
        )
    else:
        raise ValueError(
            f"Not able to convert: {train_env.action_space} -> {Discrete(n_actions)}"
        )

    eval_env = EvalGymSuite(train_env, n_episodes=10)
    train_env = SymLogWrapper(train_env)

    # build algorithm
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    subfolder_name = gym_env.split("-")[0]
    log_dir = log_dir.parent / subfolder_name / log_dir.name
    logger = [TensorBoardLogger(log_dir), CSVLogger(log_dir)]

    dyna = ERDynaQ(
        env=train_env,
        eval_env=[eval_env],
        logger=logger,
        log_dir=log_dir,
        device=device,
        **config.Dyna.to_container(),
    )
    dyna.to(device)

    if not quiet:
        print("Train on environment: ", gym_env)
        print(config)
        print(dyna)

    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", "yes"]):
            print("Abort training")
            return

    # train algorithm
    dyna.train(config.episode_budget, verbose=not quiet)


def train_md_dyna_gym_env(
    config: DictConfig,
    force: bool,
    gym_env: str,
    max_steps: int = 200,
    n_actions: int = 10,
    device: str = "cpu",
    quiet: bool = False,
):
    config: MDDynaConfig = MDDynaConfig.from_dict_config(config)
    # build envs (train, eval env)
    kwargs = {}
    if "LunarLander" in gym_env:
        kwargs["continuous"] = True
    train_env = gymnasium.make(gym_env, max_episode_steps=max_steps, **kwargs)
    if isinstance(train_env.action_space, Box):
        assert (
            len(train_env.action_space.shape) > 0
        ), "For Multidiscrete Dyna you need a one dimensional action space"
        nvec = np.ones(train_env.action_space.shape) * n_actions
        train_env = Box2MultiDiscreteActionWrapper(train_env, nvec)
    else:
        assert isinstance(
            train_env.action_space, MultiDiscrete
        ), "asserted MultiDiscrete action space in  given environment"
        logging.warning(
            f"Argument n_actions will be ignored. Instead use nvec from given actions space: n={train_env.action_space.nvec}"
        )

    eval_env = EvalGymSuite(train_env, n_episodes=10)
    # train_env = SymLogWrapper(train_env)

    # build algorithm
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    subfolder_name = gym_env.split("-")[0]
    log_dir = log_dir.parent / subfolder_name / log_dir.name
    logger = [TensorBoardLogger(log_dir), CSVLogger(log_dir)]

    dyna = MultiDiscreteDynaQ(
        env=train_env,
        eval_env=[eval_env],
        logger=logger,
        log_dir=log_dir,
        device=device,
        **config.Dyna.to_container(),
    )
    dyna.to(device)

    if not quiet:
        print("Train on environment: ", gym_env)
        print(config)
        print(dyna)

    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", "yes"]):
            print("Abort training")
            return

    # train algorithm
    dyna.train(config.episode_budget, verbose=not quiet)
