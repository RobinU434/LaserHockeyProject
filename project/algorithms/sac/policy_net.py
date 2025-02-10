from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.distributions import Normal as _Normal
from torch.distributions import Beta as _Beta

from project.algorithms.sac.actor import Actor
from project.algorithms.sac.q_net import QNet
from project.algorithms.utils import get_min_q

class Beta(_Beta):
    def __init__(self, log_concentration1, log_concentration0, validate_args=None):
        concentration1 = F.softplus(log_concentration1)
        concentration0 = F.softplus(log_concentration0)
        super().__init__(concentration1, concentration0, validate_args)

    @classmethod
    def from_stats(cls, loc: Tensor, log_scale: Tensor) -> "Beta":
        scale = F.softplus(log_scale)
        loc = torch.sigmoid(loc) # has to be between 0 and 1
        alpha = ((1 - loc) / (scale * scale) - 1 / loc) * loc * loc
        beta  = alpha * (1 / loc - 1)
        return cls(alpha, beta)
    
    def rsample(self, sample_shape = ...):
        sample = super().rsample(sample_shape)
        return sample
    
    def squish(self, sample: Tensor) -> Tensor:
        """to interval: -1, 1

        Args:
            sample (_type_): _description_

        Returns:
            _type_: _description_
        """
        return sample * 2 - 1
    

class Normal(_Normal):
    def __init__(self, loc, log_scale, validate_args=None):
        scale = F.softplus(log_scale)
        super().__init__(loc, scale, validate_args)

    def squish(self, sample: Tensor) -> Tensor:
        """to interval -1, 1

        Args:
            sample (_type_): _description_

        Returns:
            _type_: _description_
        """
        return torch.tanh(sample)

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
        action_scale: float = None,  # span with of action box (max_action - min_action)
        action_bias: float = 0,  # lower end of the action box (center of action box)
    ):
        super(PolicyNet, self).__init__()
        self.action_scale = action_scale
        self.action_bias = action_bias
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
        self.log_alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr_alpha)

    def forward(self, x: torch.Tensor, mode: bool = False) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): shape = (batch_size, input_size)

        Returns:
            Tuple(torch.tensor):
        """
        param_1, param_2 = self.actor.forward(x)
        # std = torch.exp(log_std)
        # if mode = True -> sample from mode else sample with respect to a distribution with non zero std
        # std = std * (1.0 - mode)

        # distribution = Normal(param_1, param_2)
        distribution = Beta(param_1, param_2)
        
        action = distribution.rsample()
        log_prob: Tensor = distribution.log_prob(action)
        action = action # transform into [-1, 1]
        # action = sample * std + mu
        # sum log prob
        # independence assumption between individual probabilities
        # log(p(a1, a2)) = log(p(a1) * p(a2)) = log(p(a1)) + log(p(a2)
        if len(log_prob.shape) > 1 and log_prob.shape[1] > 1:
            log_prob = log_prob.sum(dim=1, keepdim=True)
        # else:
        #     log_prob = log_prob[:, 0]

        # real_action = (torch.tanh(action) + 1.0) * torch.pi  # multiply by pi in order to match the action space

        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is a bijection and differentiable
        # this equation can be found in the original paper as equation (21)
        # sum over action dimension
        real_log_prob = log_prob - torch.sum(
            torch.log(1 - torch.tanh(action).square() + 1e-7), dim=-1, keepdim=True
        )

        # TODO(RobunU434): add post processor functionality in here
        if self.action_scale is not None:
            action = distribution.squish(action) * self.action_scale
        action = action + self.action_bias
        return action, real_log_prob

    def train_net(
        self,
        q1: QNet,
        q2: QNet,
        mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        target_entropy: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        # minus: to change the sign from the log prob -> log prob is normally negative
        entropy = -self.log_alpha.exp().detach() * log_prob

        min_q = get_min_q(q1, q2, s, a)

        loss = -min_q - entropy  # for gradient ascent
        loss = torch.mean(loss)
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
