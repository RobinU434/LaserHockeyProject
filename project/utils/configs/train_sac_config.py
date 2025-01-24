from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class _Env(StructuredConfig):
    opponent: str = None
    self_play_period: int = None


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
    action_magnitude: int = None


@dataclass
class _Actor(StructuredConfig):
    architecture: list = None
    activation_function: str = None
    latent_dim: int = None


@dataclass
class Config(StructuredConfig):
    results_dir: str = None
    Env: _Env = None
    SAC: _SAC = None
    actor: _Actor = None

    def __post_init__(self):
        self.Env = _Env(**self.Env)  #pylint: disable=E1134
        self.SAC = _SAC(**self.SAC)  #pylint: disable=E1134
        self.actor = _Actor(**self.actor)  #pylint: disable=E1134
