from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal, constraints

from project.algorithms.sac.actor import Actor


class PolicyNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 1e-4,
        init_alpha: float = 0.01,
        lr_alpha: float = 1e-4,
        architecture: List[int] = [128, 128],
        latent_dim: int = 128,
        activation_function: str = "ReLU",
        action_magnitude: float = None,
    ):
        super(PolicyNet, self).__init__()
        self.actor = Actor(
            input_dim=input_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            architecture=architecture,
            latent_dim=latent_dim,
            activation_function=activation_function,
        )
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.action_magnitude = action_magnitude
        self.normal_distr = Normal(
            torch.zeros(output_dim),
            torch.ones(output_dim),
            validate_args={"scale": constraints.greater_than_eq},
        )

    def forward(self, x: torch.Tensor, mode: bool = False) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): shape = (batch_size, input_size)

        Returns:
            Tuple(torch.tensor):
        """
        mu, std = self.actor.forward(x)

        # if mode = True -> sample from mode else sample with respect to a distribution with non zero std
        std = std * (1 - mode)

        sample = torch.randn(size=mu.shape)
        log_prob = self.normal_distr.log_prob(sample)

        action = sample * std + mu
        # sum log prob
        # independence assumption between individual probabilities
        # log(p(a1, a2)) = log(p(a1) * p(a2)) = log(p(a1)) + log(p(a2)
        if log_prob.shape[1] > 1:
            log_prob = log_prob.sum(dim=1)
        else:
            log_prob = log_prob[:, 0]

        # real_action = (torch.tanh(action) + 1.0) * torch.pi  # multiply by pi in order to match the action space
        # TODO(RobunU434): add post processor functionality in here
        if self.action_magnitude is not None:
            action = torch.tanh(action) * self.action_magnitude

        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is a bijection and differentiable
        # this equation can be found in the original paper as equation (21)
        real_log_prob = log_prob - torch.sum(
            torch.log(1 - torch.tanh(action).pow(2) + 1e-7), dim=-1
        )  # sum over action dimension

        return action, real_log_prob

    def train_net(self, q1, q2, mini_batch, target_entropy):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        # minus: to change the sign from the log prob -> log prob is normally negative
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = (-min_q - entropy).mean()  # for gradient ascent
        self.actor.train(loss)

        # learn alpha parameter
        self.log_alpha_optimizer.zero_grad()
        # if log_prob + (-target_entropy) is positive -> make log_alpha as big as positive
        # if log_prob + (-target_entropy) is negative -> make log_alpha as small as positive
        alpha_loss = -(
            self.log_alpha.exp() * (log_prob + target_entropy).detach()
        ).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return entropy, loss, alpha_loss

    @property
    def optimizer(self):
        return self.actor.optimizer
