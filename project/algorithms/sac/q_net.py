from typing import Tuple
import torch.nn.functional as F
from torch import Tensor, optim

from project.algorithms.common.q_net import VectorizedQNet


class QNet(VectorizedQNet):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        learning_rate: float = 1e-3,
    ):
        super().__init__(state_dim, action_dim, latent_dim, [128], [128], [64, 32])
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

    def train_net(
        self, target: Tensor, mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tensor:
        s, a, _, _, _ = mini_batch
        q_val = self.forward(s, a)
        loss = F.smooth_l1_loss(q_val, target).mean()
        # loss = self.criterion.forward(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
