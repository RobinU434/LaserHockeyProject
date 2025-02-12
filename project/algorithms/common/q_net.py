from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from project.algorithms.common.network import FeedForwardNetwork


class _QNet(nn.Module):
    def soft_update(self, net_target: nn.Module, tau: float):
        """update the given parameters with the parameters from the module 

        Args:
            net_target (nn.Module): parameters to update
            tau (float): update strength (value between 0 and 1)
        """
        assert tau >= 0 and tau <= 1, f"tau has to be between 0 and 1, given value: {tau}"
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class VectorizedQNet(_QNet):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        action_head_architecture: List[int] = [128],
        state_head_architecture: List[int] = [128],
        latent_mlp_architecture: List[int] = [64, 32],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.action_head = FeedForwardNetwork(
            action_dim,
            latent_dim,
            architecture=action_head_architecture,
            activation_function="ReLU",
            final_activation="ReLU",
        )
        self.state_head = FeedForwardNetwork(
            state_dim,
            latent_dim,
            architecture=state_head_architecture,
            activation_function="ReLU",
            final_activation="ReLU",
        )
        self.latent_mlp = FeedForwardNetwork(
            2 * latent_dim,
            1,
            architecture=latent_mlp_architecture,
            activation_function="ReLU",
            final_activation=None,
        )

    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        h_state = self.state_head.forward(s)
        h_action = self.action_head.forward(a)

        latent = torch.cat([h_state, h_action], dim=1)
        q = self.latent_mlp.forward(latent)
        return q


class DiscreteQNet(_QNet):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        architecture: List[int] = [128],
        activation_function: str = "ReLU",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.net = FeedForwardNetwork(
            state_dim, n_actions, architecture, activation_function
        )

    def forward(self, state: Tensor) -> Tensor:
        """_summary_

        Args:
            state (Tensor): (batch_size, action_dim)

        Returns:
            Tensor: (batch_size, n_actions)
        """
        return self.net.forward(state)

    def action_probs(self, state: Tensor) -> Tensor:
        return F.softmax(self.forward(state))

    def action_log_probs(self, state: Tensor) -> Tensor:
        return F.log_softmax(self.forward(state))


class VariationalDiscreteQNet(DiscreteQNet):
    def __init__(
        self,
        state_dim,
        n_actions,
        architecture=[128],
        activation_function="ReLU",
        *args,
        **kwargs
    ):
        super().__init__(
            state_dim, n_actions, architecture, activation_function, *args, **kwargs
        )
        raise NotImplementedError
