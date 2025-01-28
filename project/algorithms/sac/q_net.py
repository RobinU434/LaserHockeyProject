import torch
import torch.nn.functional as F
from torch import nn, optim

from project.algorithms.common.network import FeedForwardNetwork


class QNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        learning_rate: float = 1e-3,
    ):
        super(QNet, self).__init__()

        self.action_head = FeedForwardNetwork(
            action_dim,
            latent_dim,
            architecture=[128],
            activation_function="ReLU",
            final_activation="ReLU",
        )
        self.state_head = FeedForwardNetwork(
            state_dim,
            latent_dim,
            architecture=[128],
            activation_function="ReLU",
            final_activation="ReLU",
        )
        self.latent_mlp = FeedForwardNetwork(
            2 * latent_dim,
            1,
            architecture=[128, 32],
            activation_function="ReLU",
            final_activation="ReLU",
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, s, a):
        h_state = self.state_head.forward(s)
        h_action = self.action_head.forward(a)

        latent = torch.cat([h_state, h_action], dim=1)
        q = self.latent_mlp.forward(latent)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, _, _ = mini_batch
        q_val = self.forward(s, a)
        loss = F.smooth_l1_loss(q_val, target).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def soft_update(self, net_target, tau):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
