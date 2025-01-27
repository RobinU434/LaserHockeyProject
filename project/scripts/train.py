import hydra
from omegaconf import DictConfig
from stable_baselines3.common.logger import configure

from project.algorithms.logger import CSVLogger, TensorBoardLogger
from project.algorithms.sac.sac import SAC
from project.algorithms.trainer import (ExponentialSampler, SelfPlayTrainer,
                                        WarmupSchedule)
from project.environment.evaluate_env import EvalHockeEnv
from project.environment.hockey_env.hockey.hockey_env import HockeyEnv
from project.environment.single_player_env import SinglePlayerHockeyEnv
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


def train_sac(config: DictConfig, force: bool = False):
    config: SACConfig = SACConfig.from_dict_config(config)
    # build envs (train, eval env)
    train_env = SinglePlayerHockeyEnv(**config.SelfPlay.Env.to_container())
    eval_env = EvalHockeEnv(**config.SelfPlay.Env.to_container())

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
        **config.SAC.to_container()
    )

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
