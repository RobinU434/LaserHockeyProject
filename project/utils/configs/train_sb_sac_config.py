from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class _Env(StructuredConfig):
    keep_mode: bool = None
    mode: str = None
    verbose: bool = None


@dataclass
class _WarmupSchedule(StructuredConfig):
    n_episodes_weak: int = None
    n_episodes_strong: int = None


@dataclass
class _SelfPlay(StructuredConfig):
    self_play_period: int = None
    opponent: str = None
    Env: _Env = None
    WarmupSchedule: _WarmupSchedule = None

    def __post_init__(self):
        self.Env = _Env(**self.Env)  # pylint: disable=E1134
        self.WarmupSchedule = _WarmupSchedule(
            **self.WarmupSchedule
        )  # pylint: disable=E1134


@dataclass
class _SAC(StructuredConfig):
    learning_rate: float = None
    buffer_size: int = None
    learning_starts: int = None
    batch_size: int = None
    tau: float = None
    gamma: float = None
    train_freq: int = None
    gradient_steps: int = None
    action_noise: NoneType = None
    optimize_memory_usage: bool = None
    ent_coef: str = None
    target_update_interval: int = None
    target_entropy: str = None
    use_sde: bool = None
    sde_sample_freq: int = None
    use_sde_at_warmup: bool = None
    stats_window_size: int = None


@dataclass
class Config(StructuredConfig):
    results_dir: str = None
    episode_budget: int = None
    eval_check_interval: int = None
    save_interval: int = None
    SelfPlay: _SelfPlay = None
    SAC: _SAC = None

    def __post_init__(self):
        self.SelfPlay = _SelfPlay(**self.SelfPlay)  # pylint: disable=E1134
        self.SAC = _SAC(**self.SAC)  # pylint: disable=E1134
