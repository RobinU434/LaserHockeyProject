from typing import Dict, Tuple
from project.algorithms.common.q_net import DiscreteQNet
from project.algorithms.common.q_net import MultiDiscreteQNet as _MultiDiscreteQNet
import torch
from torch import Tensor
from torch import nn


class QNet(DiscreteQNet):
    def __init__(
        self,
        state_dim,
        n_actions,
        architecture=[],
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

    def train_net(
        self, mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], target: Tensor
    ) -> Dict[str, float]:
        state, action, _, _, _ = mini_batch

        action = action.int()
        if action.shape[1] == 1:
            action = action[:, 0]
        
        prediction = self.forward(state, action)
        # prediction = prediction[torch.arange(len(action)), action]
        loss = self.criterion.forward(prediction, target[:, 0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"critic_loss": loss.item()}


class MultiDiscreteQNet(_MultiDiscreteQNet):
    def __init__(
        self,
        state_dim,
        nvec,
        latent_dim=128,
        action_head_architecture=[128],
        state_head_architecture=[128],
        latent_mlp_architecture=[64, 32],
        memory_optimization=False,
        lr: float =  1e-4,
        *args,
        **kwargs
    ):
        super().__init__(
            state_dim,
            nvec,
            latent_dim,
            action_head_architecture,
            state_head_architecture,
            latent_mlp_architecture,
            memory_optimization,
            *args,
            **kwargs
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def train_net(
        self, mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], target: Tensor
    ) -> Dict[str, float]:
        state, action, _, _, _ = mini_batch

        action = action.int()
        if action.shape[1] == 1:
            action = action[:, 0]
            
        # get only diagonal values because  of one to one relation
        prediction = self.one2one_forward(state, action)
        # print(prediction.shape, target.shape)
        loss = self.criterion.forward(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"critic_loss": loss.item()}
