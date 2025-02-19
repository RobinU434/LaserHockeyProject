from itertools import product
from typing import Dict, List, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from project.algorithms.common.network import FeedForwardNetwork
from project.algorithms.utils.encoding import multi_hot


class _QNet(nn.Module):
    def __init__(self, device: str | torch.device = "cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

    def soft_update(self, net_target: nn.Module, tau: float):
        """update the given parameters with the parameters from the module

        Args:
            net_target (nn.Module): parameters to update
            tau (float): update strength (value between 0 and 1)
        """
        assert (
            tau >= 0 and tau <= 1
        ), f"tau has to be between 0 and 1, given value: {tau}"
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    def train_net(self, *args, **kwargs) -> Dict[str, float]:
        """hosts logic to train the QNEet

        Returns:
            Dict[str, float]: information about the training stats
        """
        raise NotImplementedError

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """evaluate the how good an action in a given state state is

        Args:
            state (Tensor): state description ((batch_dim), state_dim)
            action (Tensor): action description ((batch_dim), action_dim)


        Returns:
            Tensor: ((batch_dim), 1)
        """
        raise NotImplementedError


class VectorizedQNet(_QNet):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        action_head_architecture: List[int] = [128],
        state_head_architecture: List[int] = [128],
        latent_mlp_architecture: List[int] = [64, 32],
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(device, *args, **kwargs)

        self.action_head = FeedForwardNetwork(
            action_dim,
            latent_dim,
            architecture=action_head_architecture,
            activation_function="ReLU",
            final_activation="ReLU",
            device=self._device,
        )

        self.state_head = FeedForwardNetwork(
            state_dim,
            latent_dim,
            architecture=state_head_architecture,
            activation_function="ReLU",
            final_activation="ReLU",
            device=self._device,
        )
        self.latent_mlp = FeedForwardNetwork(
            2 * latent_dim,
            1,
            architecture=latent_mlp_architecture,
            activation_function="ReLU",
            final_activation=None,
            device=self._device,
        )

    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        h_state = self.state_head.forward(s)
        h_action = self.action_head.forward(a)

        latent = torch.cat([h_state, h_action], dim=-1)
        q = self.latent_mlp.forward(latent)
        return q


class DiscreteQNet(_QNet):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        architecture: List[int] = [128],
        activation_function: str = "ReLU",
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(device, *args, **kwargs)
        self.net = FeedForwardNetwork(
            state_dim,
            n_actions,
            architecture,
            activation_function,
            device=self._device,
        )

    def complete_forward(self, state: Tensor) -> Tensor:
        """evaluate at for every action

        Args:
            state (Tensor): (batch_size, state_dim)

        Returns:
            Tensor: (batch_size, n_actions)
        """
        return self.net.forward(state)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """evaluate at a certain action

        Args:
            state (Tensor): (batch_size, state_dim)
            action (Tensor): index of action you would like to evaluate (batch_size)

        Returns:
            Tensor: (batch_size, )
        """

        q_val = self.complete_forward(state)
        indices = torch.arange(len(q_val))
        q_val = q_val[indices, action]
        return q_val

    def action_probs(self, state: Tensor) -> Tensor:
        """softmax over the last dimension

        Args:
            state (Tensor): (batch_size, state_dim)

        Returns:
            Tensor: (batch_size, n_actions)
        """
        return F.softmax(self.complete_forward(state), dim=-1)

    def action_log_probs(self, state: Tensor) -> Tensor:
        """log softmax over the last dimension

        Args:
            state (Tensor): (batch_size, state_dim)

        Returns:
            Tensor: (batch_size, n_actions)
        """
        return F.log_softmax(self.complete_forward(state), dim=-1)


class MultiDiscreteQNet(VectorizedQNet):
    def __init__(
        self,
        state_dim,
        nvec: np.ndarray,
        latent_dim=128,
        action_head_architecture=[128],
        state_head_architecture=[128],
        latent_mlp_architecture=[64, 32],
        memory_optimization: bool = False,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ):
        self.nvec = nvec
        action_dim = self.nvec.sum()
        super().__init__(
            state_dim,
            action_dim,
            latent_dim,
            action_head_architecture,
            state_head_architecture,
            latent_mlp_architecture,
            device,
            *args,
            **kwargs,
        )
        self.memory_optimization = memory_optimization

        grids = np.meshgrid(*[range(i) for i in self.nvec], indexing="ij")
        self._all_actions = np.vstack(list(map(np.ravel, grids))).T

    def forward(self, state, action):
        if len(action.shape) == 1:
            action = self._encode_actions(action[None])[0]
        else:
            action = self._encode_actions(action)

        if len(action.shape) == 1 and len(state.shape) == 2:
            action = action[None]
        elif len(action.shape) == 2 and len(state.shape) == 1:
            state = state[None]

        return super().forward(state, action)

    def _encode_actions(self, actions: Tensor) -> Tensor:
        enc = multi_hot(actions.cpu(), torch.from_numpy(self.nvec)).float()
        enc = enc.to(actions.device)
        return enc

    def pair_forward(self, state: Tensor, actions: Tensor) -> Tensor:
        """

        Args:
            state (Tensor): (state_batch_dim, state_dim)
            actions (Tensor): (action_batch_dim, action_dim), action dim has the same length as nvec and is in index space

        Returns:
            Tensor: (state_batch_dim, action_batch_dim, 1)
        """
        assert (
            len(state.shape) == 2
        ), f"expected (batch_size, state_dim) got: {state.shape}"
        assert (
            len(actions.shape) == 2
        ), f"expected (batch_size, actions_dim) got: {actions.shape}"

        n_states = len(state)
        n_actions = len(actions)

        if not self.memory_optimization:
            # batch forward
            actions = self._encode_actions(actions)
            actions = actions[None].repeat(n_states, 1, 1)
            state = state[None].repeat(n_actions, 1, 1)
            state = einops.rearrange(state, "A S F -> S A F")
            out = super().forward(state, actions)
            return out

        res = torch.empty(n_states, n_actions, 1)
        for state_idx, action_idx in product(range(n_states), range(n_actions)):
            res[state_idx, action_idx] = super().forward(
                state[state_idx], actions[action_idx]
            )
        return res

    def one2one_forward(self, state: Tensor, actions: Tensor) -> Tensor:
        """each state has a corresponding action

        Args:
            state (Tensor): (batch_dim, state_dim)
            actions (Tensor): (batch_dim, action_dim) in index space

        Returns:
            Tensor: (batch_dim, 1)
        """
        actions = self._encode_actions(actions)
        return super().forward(state, actions)

    def mc_forward(self, state: Tensor, n_samples: int) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            state (Tensor): _description_
            n_samples (int): how many samples

        Returns:
            Tuple[Tensor, Tensor]:
                - q_val (batch_size, n_actions, 1)
                - actions (batch_size, n_actions, action_dim)
        """
        if len(self._all_actions) <= n_samples:
            # do full sweep
            return self.complete_forward(state)
        else:
            idx = np.random.choice(len(self._all_actions), size=(n_samples,))
            actions = self._all_actions[idx]

        if len(state.shape) == 1:
            state = state[None]
        batch_size = len(state)

        device = state.device
        np_actions = actions
        np_actions = np_actions[None].repeat(batch_size, axis=0)
        actions = torch.from_numpy(actions).to(device)

        return self.pair_forward(state, actions), torch.from_numpy(np_actions)

    def complete_forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            state (Tensor): ((batch_size), state_dim) if no batch_size given assume 1

        Returns:
            Tuple[Tensor, Tensor]:
                - q_values: (batch_size, n_all_actions, 1)
                - actions: (batch_size, n_all_actions, action_dim)
        """
        if len(state.shape) == 1:
            state = state[None]

        # all_actions = self._encode_actions(self._all_actions)
        device = state.device
        out = self.pair_forward(state, torch.from_numpy(self._all_actions).to(device))
        all_actions = torch.from_numpy(self._all_actions)[None].repeat(len(state), 1, 1)
        return out, all_actions


class VariationalDiscreteQNet(DiscreteQNet):
    def __init__(
        self,
        state_dim,
        n_actions,
        architecture=[128],
        activation_function="ReLU",
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(
            state_dim,
            n_actions,
            architecture,
            activation_function,
            device,
            *args,
            **kwargs,
        )
        raise NotImplementedError
