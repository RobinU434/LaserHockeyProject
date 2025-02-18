from typing import Callable, Tuple
from torch import Tensor
import torch
import torch.nn as nn
from torch.distributions import Normal


class RecurrentModel(nn.Module):
    def __init__(
        self,
        state_repr_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        head_dim: int = 128,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.state_repr_dim = state_repr_dim
        self.action_dim = action_dim

        # TODO: its possible to do weight sharing here
        self.representation_head = nn.Linear(self.state_repr_dim, self.head_dim)
        self.action_head = nn.Linear(self.action_dim, self.head_dim)
        self.recurrent_unit = nn.GRUCell(2 * self.head_dim, self.hidden_dim)

    def forward(
        self, state_repr: Tensor, action: Tensor, hidden_state: Tensor = None
    ) -> Tensor:
        """get next hidden state

        Args:
            state_repr (Tensor): current state representation (state_batch, state_dim)
            action (Tensor): current action (action_batch, action_batch)
            hidden_state (Tensor, optional): current hidden state (state_batch, action_batch, hidden_dim). Defaults to None.

        Returns:
            Tensor: next hidden state (state_batch, action_batch, hidden_dim)
        """
        if hidden_state is None:
            hidden_state = torch.zeros(
                (len(state_repr), len(action), self.hidden_dim),
                device=state_repr.device,
            )
        z_latent = self.representation_head.forward(state_repr)
        a_latent = self.action_head.forward(action)

        z_latent = z_latent[:, None]
        a_latent = a_latent[None]
        latent = torch.cat([z_latent, a_latent], dim=2)

        hidden_state = self.recurrent_unit.forward(latent, hidden_state)
        return hidden_state


class ConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conditional_dim: int,
        head_dim: int = 128,
        latent_dim: int = 128,
        variational: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.conditional_dim = conditional_dim
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.variational = variational

        self.input_head = nn.Linear(self.input_dim, self.head_dim)
        self.condtional_head = nn.Linear(self.conditional_dim, self.head_dim)
        self.latent_mlp = nn.Linear(2 * self.head_dim, self.latent_dim)

        if self.variational:
            self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
            self.fc_std = nn.Linear(self.latent_dim, self.latent_dim)
            self.normal = Normal(0, 1)

    def variational_forward(
        self, inp: Tensor, condition: Tensor
    ) -> Tuple[Tensor, Tensor]:
        z, _ = self.deterministic_forward(inp, condition)
        mu = self.fc_mu.forward(z)
        std = self.fc_std.forward(z)
        return mu, std

    def deterministic_forward(
        self, inp: Tensor, condition: Tensor
    ) -> Tuple[Tensor, Tensor]:
        i_latent = self.input_head.forward(inp)
        c_latent = self.condtional_head.forward(condition)
        latent = torch.cat([i_latent, c_latent], dim=1)
        z = self.latent_mlp.forward(latent)
        return z, None

    def forward(self, inp: Tensor, condition: Tensor) -> Tensor:
        if self.variational:
            mu, std = self.variational_forward(inp, condition)
            epsilon = self.normal.sample(mu.shape)
            z = epsilon * std + mu
        else:
            z, _ = self.deterministic_forward(inp, condition)
        return z


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        conditional_dim: int,
        head_dim: int = 128,
        latent_dim: int = 128,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.conditional_dim = conditional_dim
        self.head_dim = head_dim
        self.latent_dim = latent_dim

        self.conditional_head = nn.Linear(self.conditional_dim, self.head_dim)
        self.latent_mlp = nn.Linear(self.latent_dim + self.head_dim, self.output_dim)

    def forward(self, latent: Tensor, condition: Tensor) -> Tensor:
        c_latent = self.conditional_head.forward(condition)
        z = torch.cat([latent, c_latent], dim=1)
        out = self.conditional_head.forward(z)
        return out


class CAE(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        conditional_dim: int,
        latent_dim: int,
        head_dim: int = 128,
        variational: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder = ConditionalEncoder(
            inp_dim, conditional_dim, head_dim, latent_dim, variational
        )
        self.decoder = ConditionalDecoder(
            inp_dim, conditional_dim, head_dim, latent_dim
        )

        self.normal = Normal(0, 1)

    def froward(self, inp: Tensor, condition: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.variational:
            mu, std = self.encoder.variational_forward(inp, condition)
            epsilon = self.normal.sample(mu.shape)
            z = epsilon * std + mu
        else:
            mu, std = self.encoder.deterministic_forward(inp, condition)
            z = mu
        out = self.decoder.forward(z, condition)

        return out, mu, std


class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_repr_dim: int,
        hidden_dim: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_repr_dim = state_repr_dim
        self.hidden_dim = hidden_dim

        self.recurrent_unit = RecurrentModel(
            self.state_repr_dim, self.action_dim, self.hidden_dim, head_dim=128
        )
        self.repr_model = ConditionalEncoder(self.obs_dim, self.hidden_dim)  # encoder

        self.transition_model = nn.Linear(self.hidden_dim, self.state_repr_dim)

    def forward(
        self, observation: Tensor, action: Tensor, hidden_state: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """get state representation and new hidden state

        Args:
            observation (Tensor): observation you took in a given state (obs_batch, obs_dim)
            action (Tensor): action to execute for a state with the given observation (action_batch, action_dim)
            hidden_state (Tensor, optional): Defaults to None (obs_batch, action_batch, hidden_state)

        Returns:
            Tuple[Tensor, Tensor]: state representation, estimated_state_repr, next hidden state
        """
        if hidden_state is None:
            obs_batch = len(observation)
            action_batch = len(action)
            hidden_state = torch.ones(
                (obs_batch, action_batch, self.hidden_dim), device=observation.device
            )
        state_repr = self.repr_model.forward(observation, hidden_state)
        next_hidden_state = self.recurrent_unit.forward(
            state_repr, action, hidden_state
        )
        state_repr_hat = self.transition_model.forward(hidden_state)

        return
