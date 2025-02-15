from typing import Dict, List, Tuple
from torch import Tensor, nn
import torch
from project.algorithms.common.network import FeedForwardNetwork
from torch.nn import Sequential, Softmax


class WorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        architecture: List[int] = [128, 128],
        activation_func: str = "ReLU",
        lr: float = 1e-4,
    ):
        super(WorldModel, self).__init__()
        # build network layers
        self.state_head = FeedForwardNetwork(state_dim, latent_dim, architecture=[128])
        self.action_head = FeedForwardNetwork(
            action_dim, latent_dim, architecture=[128]
        )
        self.hidden_model = FeedForwardNetwork(
            2 * latent_dim,
            latent_dim,
            architecture=architecture,
            activation_function=activation_func,
        )

        self._next_state_pred = FeedForwardNetwork(
            latent_dim, state_dim, architecture=[128]
        )
        self._reward_pred = FeedForwardNetwork(latent_dim, 1, architecture=[128])
        self._done_pred = Sequential(
            FeedForwardNetwork(latent_dim, 1, architecture=[128]), Softmax(dim=-1)
        )

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """predict next state, reward and of the episode is done

        Args:
            state (Tensor): state repr
            action (Tensor): action repr

        Returns:
            Tuple[Tensor, Tensor, Tensor]: next_state, reward, done
        """
        action_hidden = self.action_head.forward(action)
        state_hidden = self.state_head.forward(state)
        hidden = torch.cat([action_hidden, state_hidden], dim=-1)
        hidden = self.hidden_model.forward(hidden)

        next_state = self._next_state_pred.forward(hidden)
        reward = self._reward_pred.forward(hidden)
        done = self._done_pred.forward(hidden)
        return next_state, reward, done

    def train_net(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> Dict[str, float]:
        next_state_pred, reward_pred, done_pred = self.forward(state, action)
        state_loss = self.mse.forward(next_state_pred, next_state)
        reward_loss = self.mse.forward(reward_pred, reward)
        done_loss = self.bce.forward(done_pred, done)
        loss = state_loss + reward_loss + done_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "state_loss": state_loss.item(),
            "reward_loss": reward_loss.item(),
            "done_loss": done_loss.item(),
        }
