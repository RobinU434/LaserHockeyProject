import gymnasium
import hydra
from omegaconf import DictConfig
from stable_baselines3.common.logger import configure

from project.algorithms.common.logger import CSVLogger, TensorBoardLogger
from project.algorithms.sac.sac import SAC
from project.algorithms.trainer import (ExponentialSampler, SelfPlayTrainer,
                                        WarmupSchedule)
from project.environment.evaluate_env import EvalHockeEnv
from project.environment.hockey_env.hockey.hockey_env import HockeyEnv
from project.environment.single_player_env import SinglePlayerHockeyEnv
from project.utils.configs.train_sac_config import Config as SACConfig
from project.environment.hockey_env.hdf5_replay_buffer import HDF5ReplayBuffer



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

    # ititialize HDF5ReplayBuffer (save interactions)
    hdf5_replay_buffer = HDF5ReplayBuffer("interactions.h5", max_size=100000)

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

    # train loop, saves interactions in hdf5
    for episode in range(config.episode_budget):
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

    trainer.train(config.episode_budget)
    # train algorithm