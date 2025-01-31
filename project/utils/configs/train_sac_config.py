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
class _Actor_config(StructuredConfig):
    architecture: list = None
    activation_function: str = None
    latent_dim: int = None


@dataclass
class _SAC(StructuredConfig):
    lr_pi: float = None
    lr_q: float = None
    init_alpha: float = None
    gamma: float = None
    batch_size: int = None
    buffer_limit: int = None
    start_buffer_size: int = None
    train_iterations: int = None
    tau: float = None
    target_entropy: float = None
    lr_alpha: float = None
    action_scale: int = None
    action_bias: int = None
    experience_replay: bool = None
    actor_config: _Actor_config = None

    def __post_init__(self):
        self.actor_config = _Actor_config(**self.actor_config)  # pylint: disable=E1134


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
