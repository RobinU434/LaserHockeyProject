from typing import Dict, Tuple
from project.algorithms.common.q_net import DiscreteQNet
import torch
from torch import Tensor
from torch import nn


class QNet(DiscreteQNet):
    def __init__(
        self,
        state_dim,
        n_actions,
        architecture=...,
        activation_function="ReLU",
        lr: float = 1e-4,
        *args,
        **kwargs
    ):
        super().__init__(
            state_dim, n_actions, architecture, activation_function, *args, **kwargs
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def train(
        self, mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], target: Tensor
    ) -> Dict[str, float]:
        state, action, _, _, _ = mini_batch
        prediction = self.forward(state)[:, action]

        loss = self.criterion.forward(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"critic_loss": loss.item()}
    
    def soft_update(self, net_target: nn.Module, tau: float):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

