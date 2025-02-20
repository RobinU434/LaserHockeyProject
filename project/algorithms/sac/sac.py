import math
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import Env
from gymnasium.spaces import Box
from torch import Tensor
from tqdm import tqdm

from project.algorithms.common.agent import _Agent
from project.algorithms.common.algorithm import _RLAlgorithm
from project.algorithms.common.buffer import ReplayBuffer, _ReplayBuffer

# from project.algorithms.sac.buffer import ReplayBuffer
from project.algorithms.sac.policy_net import PolicyNet
from project.algorithms.sac.q_net import QNet
from project.algorithms.sac.utils import get_min_q
from project.algorithms.utils.filesystem import get_save_path
from project.algorithms.utils.gym_helper import ContinuousPlaceHolderEnv, get_space_dim
from project.algorithms.utils.str_ops import generate_separator
from project.algorithms.utils.torch_ops import state_dict_to_cpu


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
        target_entropy: float = None,  # for automated alpha update,
        lr_alpha: float = 0.001,  # for automated alpha update
        actor_config: dict = None,
        eval_env: List[Env] = None,
        eval_check_interval: int = None,
        save_interval: int = None,
        experience_replay: bool = False,
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
        self._target_entropy = (
            target_entropy
            if target_entropy is not None
            else -float(self._env.action_space.shape.item())
        )
        self._lr_alpha = lr_alpha  # for automated alpha update
        self._actor_config = actor_config if actor_config is not None else {}
        self._experience_replay = experience_replay

        self._state_dim = get_space_dim(self._env.observation_space)
        self._action_dim = get_space_dim(self._env.action_space)

        self._memory: _ReplayBuffer
        self._build_buffer()

        self._q1: QNet
        self._q2: QNet
        self._q1_target: QNet
        self._q2_target: QNet
        self._build_q_networks()

        self._action_scale: Tensor
        self._action_bias: Tensor
        self._pi: PolicyNet
        self._build_policy()

    def _build_q_networks(self):
        self._q1 = QNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            learning_rate=self._lr_q,
            latent_dim=128,
        )
        self._q2 = QNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            learning_rate=self._lr_q,
            latent_dim=128,
        )
        self._q1_target = QNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            learning_rate=self._lr_q,
            latent_dim=128,
        )
        self._q2_target = QNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            learning_rate=self._lr_q,
            latent_dim=128,
        )

        self._q1_target.load_state_dict(self._q1.state_dict().copy())
        self._q2_target.load_state_dict(self._q2.state_dict().copy())

    def _build_policy(self):
        if not isinstance(self._env.action_space, Box):
            raise ValueError("SAC currently only works on a continuos acition space")
        action_space: Box = self._env.action_space
        self._action_scale = action_space.high - action_space.low
        self._action_scale = (
            torch.Tensor([self._action_scale])
            if isinstance(self._action_scale, float)
            else torch.from_numpy(self._action_scale)
        )
        self._action_bias = (action_space.high + action_space.low) / 2.0
        self._action_bias = (
            torch.Tensor([self._action_bias])
            if isinstance(self._action_bias, float)
            else torch.from_numpy(self._action_bias)
        )

        self._pi = PolicyNet(
            input_dim=self._state_dim,
            output_dim=self._action_dim,
            learning_rate=self._lr_pi,
            init_alpha=self._init_alpha,
            lr_alpha=self._lr_alpha,
            action_scale=self._action_scale,
            action_bias=self._action_bias,
            **self._actor_config,
        )

    def _build_buffer(self):
        # self._memory = RigidReplayBuffer(
        #     self._buffer_limit, self._action_dim, self._state_dim, 1
        # )
        self._memory = ReplayBuffer(self._buffer_limit)

    def calc_target(
        self, mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tensor:
        """compute td target

        Args:
            mini_batch (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): minibatch with:
                - state
                - action
                - next_state,
                - reward
                - done

        Returns:
            Tensor: td target (batch_size,)
        """
        _, _, s_prime, r, done = mini_batch
        s_prime = s_prime.to(self._device).float()
        r = r.to(self._device).float()
        done = done.to(self._device).float()

        with torch.no_grad():
            a_prime, log_prob = self._pi.forward(s_prime)
            entropy = -self._pi.log_alpha.exp().detach() * log_prob

            min_q = get_min_q(self._q1_target, self._q2_target, s_prime, a_prime)
            target = r + self._gamma * (1 - done) * (min_q + entropy)

        return target

    def collect_episode(self, episode_idx: int):
        step_counter = 0
        score = 0.0
        log_densities = []

        s, _ = self._env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            s_input = torch.from_numpy(s).float().to(self._device)
            # introduce batch size 1
            a, log_prob = self._pi.forward(s_input[None])
            a = a.cpu()
            log_prob = log_prob.cpu()

            # detach grad from action to apply it to the environment where it is converted into a numpy.ndarray
            a = a.detach()[0]
            s_prime, r, done, truncated, _ = self._env.step(a.numpy())
            score += r

            # NOTE: this is experimental
            # r = math.tanh(r * 0.5)

            sampling_weight = None
            # compute experience replay weight as the td error
            if self._experience_replay:
                with torch.no_grad():
                    minibatch = (
                        None,
                        None,
                        torch.from_numpy(s_prime)[None],
                        torch.tensor([r])[None],
                        torch.tensor([done], dtype=float)[None],
                    )
                    target = self.calc_target(minibatch)
                    q_pred = get_min_q(
                        self._q1,
                        self._q2,
                        torch.from_numpy(s)[None].to(self._device),
                        a[None],
                    )
                    sampling_weight = F.smooth_l1_loss(q_pred, target).mean()
                    sampling_weight = sampling_weight.cpu().item()

            self._memory.put(
                observation=torch.from_numpy(s),
                action=a,
                next_observation=torch.from_numpy(s_prime),
                reward=torch.tensor([r]),
                done=torch.tensor([done], dtype=float),
                sampling_weight=sampling_weight,
            )

            s = s_prime
            step_counter += 1
            log_densities.append(log_prob.item())

        self.log_scalar("mean_score", score, episode_idx)
        self.log_scalar("episode_steps", step_counter, episode_idx)
        self.log_scalar("sac/log_prob", np.mean(log_densities), episode_idx)

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

        logging_entropy = torch.tensor(logging_entropy).mean().item()
        actor_losses = torch.tensor(actor_losses).mean().item()
        critic_losses = torch.tensor(critic_losses).mean().item()
        alpha_losses = torch.tensor(alpha_losses).mean().item()
        alpha = self._pi.log_alpha.exp().item()
        self.log_scalar("sac/entropy", logging_entropy, episode_idx)
        self.log_scalar("sac/actor_loss", actor_losses, episode_idx)
        self.log_scalar("sac/critic_loss", critic_losses, episode_idx)
        self.log_scalar("sac/alpha_loss", alpha_losses, episode_idx)
        self.log_scalar("sac/alpha", alpha, episode_idx)

    def train(self, n_episodes=1000, verbose=False):
        if isinstance(self._env, ContinuousPlaceHolderEnv):
            raise ValueError(
                "Training with PlaceHolderEnv is not possible. Please update internal environment."
            )

        iterator = range(n_episodes)
        if verbose:
            iterator = tqdm(iterator, desc="train sac", unit="episodes")

        for episode_idx in iterator:
            self.collect_episode(episode_idx)

            if len(self._memory) >= self._start_buffer_size:
                self.train_nets(episode_idx)

            self.mid_training_hooks(episode_idx)
        self.post_training_hooK(n_episodes)

    def save_checkpoint(self, episode_idx: int, path: Path | str = None):
        path = get_save_path(self._log_dir, episode_idx, path)
        content = self.get_basic_save_args(episode_idx)
        content = {
            **content,
            "pi_model_state_dict": state_dict_to_cpu(self._pi.state_dict()),
            "pi_optimizer_state_dict": state_dict_to_cpu(
                self._pi.optimizer.state_dict()
            ),
            "q1_model_state_dict": state_dict_to_cpu(self._q1.state_dict()),
            "q1_optimizer_state_dict": state_dict_to_cpu(
                self._q1.optimizer.state_dict()
            ),
            "q2_model_state_dict": state_dict_to_cpu(self._q2.state_dict()),
            "q2_optimizer_state_dict": state_dict_to_cpu(
                self._q2.optimizer.state_dict()
            ),
            "q1_target_model_state_dict": state_dict_to_cpu(
                self._q1_target.state_dict()
            ),
            "q1_target_optimizer_state_dict": state_dict_to_cpu(
                self._q1_target.optimizer.state_dict()
            ),
            "q2_target_model_state_dict": state_dict_to_cpu(
                self._q2_target.state_dict()
            ),
            "q2_target_optimizer_state_dict": state_dict_to_cpu(
                self._q2_target.optimizer.state_dict()
            ),
        }
        torch.save(content, path)

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only=False)
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
    def __init__(
        self,
        policy: PolicyNet,
        deterministic: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self._policy = policy
        self._mode = deterministic
        self._device = device

    def act(self, state):
        state = torch.from_numpy(state).float().to(self._device)
        action, _ = self._policy.forward(state[None], mode=self._mode)
        return action.detach().cpu().numpy()[0]
