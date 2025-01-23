from typing import Union
from stable_baselines3.sac import SAC
import torch

from project.hockey_env.hockey.hockey_env import HockeyEnv, Mode


class SACSelfPlay:
    def __init__(
        self,
        env_mode: str = Mode.NORMAL,
        keep_mode: bool = True,
        update_opponent_period: int = 10000,  # in steps
        opponent: object = None,  # if given play against constant player. Otherwise self play
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str = None,
        verbose: int = 0,
        seed: int = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        self.env = HockeyEnv(keep_mode=keep_mode, mode=env_mode, verbose=verbose)
        self.sac = SAC(
            env=self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        if self.opponent is None:
            self.opponent = opponent
        else:
            self.sac = SAC(
                env=self.env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                optimize_memory_usage=optimize_memory_usage,
                ent_coef=ent_coef,
                target_update_interval=target_update_interval,
                target_entropy=target_entropy,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                use_sde_at_warmup=use_sde_at_warmup,
                stats_window_size=stats_window_size,
                tensorboard_log=tensorboard_log,
                verbose=verbose,
                seed=42,
                device=device,
                _init_setup_model=_init_setup_model,
            )

        self.update_opponent_period = update_opponent_period

    def learn(self, steps: int = 100000):
        

