from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from gymnasium import Env
from torch import Tensor
from tqdm import tqdm

from project.algorithms.common.agent import _Agent
from project.algorithms.common.algorithm import _RLAlgorithm
from project.algorithms.common.buffer import _ReplayBuffer, RigidReplayBuffer
from project.algorithms.sac.policy_net import PolicyNet
from project.algorithms.sac.q_net import QNet
from project.algorithms.utils import PlaceHolderEnv, generate_separator, get_space_dim


class SAC(_RLAlgorithm):
    def __init__(
        self,
        env,
        logger=[],
        lr_pi: float = 0.0005,
        lr_q: float = 0.001,
        init_alpha: float = 0.01,
        gamma: float = 0.98,
        batch_size: float = 32,
        buffer_limit: float = 50000,
        start_buffer_size: float = 1000,
        train_iterations: float = 20,
        tau: float = 0.01,  # for target network soft update,
        target_entropy: float = -1.0,  # for automated alpha update,
        lr_alpha: float = 0.001,  # for automated alpha update
        action_magnitude: float = 1,
        actor_config: dict = None,
        eval_env: List[Env] = None,
        eval_check_interval: int = None,
        save_interval: int = None,
        log_dir: str | Path = Path("results"),
        *args,
        **kwargs,
    ):
        super().__init__(
            env,
            logger,
            eval_env,
            eval_check_interval,
            save_interval,
            log_dir,
            *args,
            **kwargs,
        )
        self.save_hyperparmeters()

        self._lr_pi = lr_pi
        self._lr_q = lr_q
        self._init_alpha = init_alpha
        self._gamma = gamma
        self._batch_size = batch_size
        self._buffer_limit = buffer_limit
        self._start_buffer_size = start_buffer_size
        self._train_iterations = train_iterations
        self._tau = tau  # for target network soft update
        self._target_entropy = target_entropy  # for automated alpha update
        self._lr_alpha = lr_alpha  # for automated alpha update
        self._action_magnitude = action_magnitude
        self._actor_config = actor_config if actor_config is not None else {}

        self._state_dim = get_space_dim(self._env.observation_space)
        self._action_dim = get_space_dim(self._env.action_space)

        self._memory: _ReplayBuffer
        self._build_buffer()

        self._q1: QNet
        self._q2: QNet
        self._q1_target: QNet
        self._q2_target: QNet
        self._build_q_networks()

        self._pi: PolicyNet
        self._build_policy()

    def _build_q_networks(self):
        self._q1 = QNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            learning_rate=self._lr_q,
        )
        self._q2 = QNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            learning_rate=self._lr_q,
        )
        self._q1_target = QNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            learning_rate=self._lr_q,
        )
        self._q2_target = QNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            learning_rate=self._lr_q,
        )

        self._q1_target.load_state_dict(self._q1.state_dict().copy())
        self._q2_target.load_state_dict(self._q2.state_dict().copy())

    def _build_policy(self):
        self._pi = PolicyNet(
            input_dim=self._state_dim,
            output_dim=self._action_dim,
            learning_rate=self._lr_pi,
            init_alpha=self._init_alpha,
            lr_alpha=self._lr_alpha,
            action_magnitude=self._action_magnitude,
            **self._actor_config,
        )

    def _build_buffer(self):
        self._memory = RigidReplayBuffer(
            self._buffer_limit, self._action_dim, self._state_dim, 1
        )

    def calc_target(
        self, mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tensor:
        """compute td target

        Args:
            mini_batch (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): minibatch with:
                - state
                - action
                - reward
                - next_state,
                - done

        Returns:
            Tensor: td target (batch_size,)
        """
        _, _, s_prime, r, done = mini_batch
        with torch.no_grad():
            a_prime, log_prob = self._pi.forward(s_prime)
            entropy = -self._pi.log_alpha.exp() * log_prob
            entropy = entropy.unsqueeze(dim=1)

            q1_val = self._q1_target(s_prime, a_prime)
            q2_val = self._q2_target(s_prime, a_prime)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)

            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = r + self._gamma * done * (min_q + entropy)
        return target

    def collect_episode(self, episode_idx: int):
        step_counter = 0
        score = 0.0
        log_densities = []

        s, _ = self._env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            s_input = torch.from_numpy(s).float()
            # introduce batch size 1
            a, log_prob = self._pi.forward(s_input[None])

            # detach grad from action to apply it to the environment where it is converted into a numpy.ndarray
            a = a[0].detach()
            s_prime, r, done, truncated, _ = self._env.step(a.numpy())

            self._memory.put(
                torch.from_numpy(s),
                a,
                torch.from_numpy(s_prime),
                torch.tensor([r]),
                torch.tensor([done]),
            )
            s = s_prime

            score += r
            step_counter += 1
            log_densities.append(log_prob.item())

        self.log_scalar("mean_score", score / step_counter, episode_idx)
        self.log_scalar("episode_steps", step_counter, episode_idx)
        self.log_scalar("sac/log_density", np.mean(log_densities), episode_idx)

    def train_nets(self, episode_idx: int):
        logging_entropy = []
        actor_losses = []
        critic_losses = []
        alpha_losses = []

        for _ in range(self._train_iterations):
            mini_batch = self._memory.sample(self._batch_size)
            td_target = self.calc_target(mini_batch)

            critic_loss = self._q1.train_net(td_target, mini_batch)
            critic_losses.append(critic_loss.item())
            critic_loss = self._q2.train_net(td_target, mini_batch)
            critic_losses.append(critic_loss.item())

            entropy, actor_loss, alpha_loss = self._pi.train_net(
                self._q1, self._q2, mini_batch, self._target_entropy
            )

            logging_entropy.append(entropy.mean())
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)

            self._q1.soft_update(self._q1_target, self._tau)
            self._q2.soft_update(self._q2_target, self._tau)

        self.log_scalar(
            "sac/entropy", torch.tensor(logging_entropy).mean().item(), episode_idx
        )
        self.log_scalar(
            "sac/actor_loss", torch.tensor(actor_losses).mean().item(), episode_idx
        )
        self.log_scalar(
            "sac/critic_loss", torch.tensor(critic_losses).mean().item(), episode_idx
        )
        self.log_scalar(
            "sac/alpha_loss", torch.tensor(alpha_losses).mean().item(), episode_idx
        )

    def train(self, n_episodes: int = 1000):
        if isinstance(self._env, PlaceHolderEnv):
            raise ValueError(
                "Training with PlaceHolderEnv is not possible. Please update internal environment."
            )
        for episode_idx in tqdm(range(n_episodes), desc="train sac", unit="episodes"):
            self.collect_episode(episode_idx)

            if len(self._memory) > self._start_buffer_size:
                self.train_nets(episode_idx)

            if (
                self._save_interval is not None
                and (n_episodes + 1) % self._save_interval == 0
            ):
                # save model
                self.save_checkpoint(episode_idx)

            if (
                self._eval_check_interval is not None
                and (n_episodes + 1) % self._eval_check_interval == 0
            ):
                self.evaluate(episode_idx)

        # store metrics in a csv file
        self.save_metrics()
        self.save_checkpoint(n_episodes)
        self._env.close()

    def save_checkpoint(self, episode_idx: int, path: Path | str = None):
        self._log_dir.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = self._log_dir / f"checkpoint_{episode_idx}.pt"

        torch.save(
            {
                "epoch": episode_idx,
                "pi_model_state_dict": self._pi.state_dict(),
                "pi_optimizer_state_dict": self._pi.optimizer.state_dict(),
                "q1_model_state_dict": self._q1.state_dict(),
                "q1_optimizer_state_dict": self._q1.optimizer.state_dict(),
                "q2_model_state_dict": self._q2.state_dict(),
                "q2_optimizer_state_dict": self._q2.optimizer.state_dict(),
                "q1_target_model_state_dict": self._q1_target.state_dict(),
                "q1_target_optimizer_state_dict": self._q1_target.optimizer.state_dict(),
                "q2_target_model_state_dict": self._q2_target.state_dict(),
                "q2_target_optimizer_state_dict": self._q2_target.optimizer.state_dict(),
                "hparams": vars(self.hparams),
                "action_dim": self._action_dim,
                "state_dim": self._state_dim,
            },
            path,
        )

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only=True)
        self._pi.load_state_dict(checkpoint["pi_model_state_dict"])
        self._pi.optimizer.load_state_dict(checkpoint["pi_optimizer_state_dict"])
        self._q1.load_state_dict(checkpoint["q1_model_state_dict"])
        self._q1.optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
        self._q2.load_state_dict(checkpoint["q2_model_state_dict"])
        self._q2.optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])
        self._q1_target.load_state_dict(checkpoint["q1_target_model_state_dict"])
        self._q1_target.optimizer.load_state_dict(
            checkpoint["q1_target_optimizer_state_dict"]
        )
        self._q2_target.load_state_dict(checkpoint["q2_target_model_state_dict"])
        self._q2_target.optimizer.load_state_dict(
            checkpoint["q2_target_optimizer_state_dict"]
        )

    def get_agent(self, deterministic: bool = False) -> _Agent:
        return SACAgent(deepcopy(self._pi), deterministic)

    def __repr__(self):
        s1 = generate_separator("Policy", 80)
        s2 = generate_separator("Q-Function", 80)
        pi_str = str(self._pi)
        q_str = str(self._q1)
        s = "\n".join([s1, pi_str, s2, q_str])
        return s


class SACAgent(_Agent):
    def __init__(self, policy: PolicyNet, deterministic: bool = False):
        super().__init__()
        self._policy = policy
        self._mode = deterministic

    def act(self, state):
        state = torch.from_numpy(state).float()
        action, _ = self._policy.forward(state[None], mode=self._mode)
        return action.detach().numpy()[0]
