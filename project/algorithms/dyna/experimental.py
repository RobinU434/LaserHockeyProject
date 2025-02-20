import numpy as np
import torch
from project.algorithms.dyna.dyna import DynaQ
from torch import optim
from torch.nn import functional as F


class ERDynaQ(DynaQ):
    """entropy regularized dyna.

    don't set a decay rate for epsilon greedy policies.
    Let the network decide how greedy it would like to be based on a target entropy

    """

    def __init__(
        self,
        env,
        logger=...,
        eval_env=None,
        eval_check_interval=None,
        save_interval=None,
        log_dir=None,
        batch_size=256,
        buffer_limit=50000,
        start_buffer_size=1000,
        target_entropy: float = None,  # 0 = point mass
        init_alpha: float = 1e-4,  # small emphasis randomness
        lr_alpha: float = 1e-3,
        gamma=0.99,
        tau=0.01,
        simulation_updates=2,
        device="cpu",
        *args,
        **kwargs
    ):
        super().__init__(
            env,
            logger,
            eval_env,
            eval_check_interval,
            save_interval,
            log_dir,
            batch_size,
            buffer_limit,
            start_buffer_size,
            0,  # not needed
            gamma,
            tau,
            simulation_updates,
            device,
            *args,
            **kwargs
        )

        self.target_entropy = target_entropy
        if self.target_entropy is None:
            self.target_entropy = np.log(self._n_actions)
        self.init_alpha = init_alpha

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr_alpha)

    def get_action(self, state, episode_idx):
        with torch.no_grad():
            q_val = self.q_net.complete_forward(state)
        
        action_log_probs = F.log_softmax(torch.exp(self.log_alpha) * q_val, dim=-1)
        action = np.random.choice(
            len(action_log_probs), p=torch.exp(action_log_probs).cpu().detach().numpy()
        )
        log_prob = action_log_probs[action].cpu().item()

        return action, log_prob
    
    def _get_epsilon(self, episode_idx):
        return torch.exp(self.log_alpha).item()

    def update_q(self, mini_batch, episode_idx):
        super().update_q(mini_batch, episode_idx)

        # optimize alpha
        alpha = torch.exp(self.log_alpha)
        state = mini_batch[0]
        state = state.to(self._device)


        q_val = self.q_net.complete_forward(state)
        action_prob = F.softmax(alpha * q_val, dim=-1)
        action_prob = action_prob + 1e-18  # avoid 0
        entropy = -torch.sum(action_prob * torch.log(action_prob))

        loss = (entropy - self.target_entropy)**2
        self.log_alpha_optimizer.zero_grad()
        loss.backward()
        self.log_alpha_optimizer.step()


        self.log_scalar(self.get_name() + "/epsilon_loss", loss.item(), episode_idx)
        self.log_scalar(self.get_name() + "/target_entropy", self.target_entropy, episode_idx)
        self.log_scalar(self.get_name() + "/entropy", entropy.detach().item(), episode_idx)
