import logging
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch as th
from torch.nn import functional as F


class ER_SAC(SAC):
    def __init__(
        self,
        policy,
        env,
        learning_rate=0.0003,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

    def get_td_target(self, new_obs, reward, dones):
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach()).detach()
        else:
            ent_coef = self.ent_coef_tensor

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(new_obs)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.critic_target(new_obs, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = reward + (1 - dones) * self.gamma * next_q_values

        return target_q_values

    def _store_transition(
        self, replay_buffer, buffer_action, new_obs, reward, dones, infos
    ):
        # get sampling weight
        new_obs_ = th.from_numpy(new_obs)
        reward_ = th.from_numpy(reward)
        dones_ = th.from_numpy(dones).float()
        curr_obs_ = th.from_numpy(self._last_obs)
        buffer_action_ = th.from_numpy(buffer_action)
        target_q_values = self.get_td_target(new_obs_, reward_, dones_)
        current_q_values = self.critic(curr_obs_, buffer_action_)
        # Compute critic loss
        sampling_weight = 0.5 * sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        sampling_weight = sampling_weight.detach().cpu().item()
        infos[0]["sampling_weight"] = sampling_weight
        
        return super()._store_transition(
            replay_buffer, buffer_action, new_obs, reward, dones, infos
        )


class ERReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        handle_timeout_termination=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.sampling_weight = np.zeros(self.buffer_size, dtype=np.float32)

        if self.optimize_memory_usage:
            logging.warning("Experience replay sampling not enabled.")

    def add(self, obs, next_obs, action, reward, done, infos):
        self.sampling_weight[self.pos] = infos[0]["sampling_weight"]
        return super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size, env=None):
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            sw = self.sampling_weight[:upper_bound]
            p = sw / sw.sum()
            batch_inds = np.random.choice(
                upper_bound, replace=True, size=batch_size, p=p
            )
            return self._get_samples(batch_inds, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`))
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)
