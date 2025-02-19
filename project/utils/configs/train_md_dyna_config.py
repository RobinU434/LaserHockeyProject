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
class _Dyna(StructuredConfig):
    batch_size: int = None
    buffer_limit: int = None
    start_buffer_size: int = None
    epsilon_decay: float = None
    gamma: float = None
    tau: float = None
    simulation_updates: int = None
    mc_sample: NoneType = None


@dataclass
class Config(StructuredConfig):
    results_dir: str = None
    episode_budget: int = None
    eval_check_interval: int = None
    save_interval: int = None
    SelfPlay: _SelfPlay = None
    Dyna: _Dyna = None

    def __post_init__(self):
        self.SelfPlay = _SelfPlay(**self.SelfPlay)  # pylint: disable=E1134
        self.Dyna = _Dyna(**self.Dyna)  # pylint: disable=E1134
