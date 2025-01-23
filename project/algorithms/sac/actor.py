import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from enum import Enum

from typing import List, Tuple

from project.algorithms.network import FeedForwardNetwork


class TrainMode(Enum):
    STATIC = 0  # no training is permitted
    FINE_TUNING = 1  # load a model and perform fine tuning
    FROM_SCRATCH = 2


class Actor(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        learning_rate,
        architecture: List[int] = [128, 128],
        latent_dim: int = 128,
        activation_function: str = "ReLU",
    ) -> None:
        super().__init__()

        self._core = FeedForwardNetwork(
            input_dim, latent_dim, architecture, activation_function
        )
        self.fc_mu = nn.Linear(latent_dim, output_dim)
        self.fc_std = nn.Sequential(nn.Linear(latent_dim, output_dim), nn.Softplus())
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """get parameterized action distribution

        Args:
            x (torch.Tensor): state (state_dim, )

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mu (action_dim, ), std (action_dim, )
        """
        latent = self._core(x)
        
        mu = self.fc_mu(latent)
        std = self.fc_std(latent)
        return mu, std

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
