from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from project.algorithms.common.network import FeedForwardNetwork


class Actor(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        learning_rate,
        architecture: List[int] = [128, 128],
        latent_dim: int = 128,
        activation_function: str = "ReLU",
        predict_cov: bool = False,
    ) -> None:
        super().__init__()

        self._core = FeedForwardNetwork(
            input_dim, latent_dim, architecture, activation_function
        )
        self.fc_mu = nn.Sequential(nn.Linear(latent_dim, output_dim))
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
