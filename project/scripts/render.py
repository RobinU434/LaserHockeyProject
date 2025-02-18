import logging
from pathlib import Path

import gymnasium
from gymnasium.spaces import Box, Discrete

from project.algorithms.dyna.dyna import DynaQ
from project.algorithms.env_wrapper import DiscreteActionWrapper
from project.algorithms.sac.sac import SAC


def render_sac(
    checkpoint: str | Path,
    gym_env: str,
    deterministic: bool = False,
    max_steps: int = 200,
):
    env = gymnasium.make(gym_env, max_episode_steps=max_steps, render_mode="human")
    if gym_env == "LunarLander-v3":
        env = gymnasium.make(
            "LunarLander-v3",
            continuous=True,
            max_episode_steps=max_steps,
            render_mode="human",
        )

    sac = SAC.from_checkpoint(checkpoint, env)
    agent = sac.get_agent(deterministic)

    done, truncated = False, False
    s, _ = env.reset()
    while not (done or truncated):
        a = agent.act(s)
        s, r, truncated, done, _ = env.step(a)
        env.render()


def render_dyna(
    checkpoint: str | Path,
    gym_env: str,
    deterministic: bool = False,
    max_steps: int = 200,
):
    dyna: DynaQ = DynaQ.from_checkpoint(checkpoint)
    env = gymnasium.make(gym_env, max_episode_steps=max_steps, render_mode="human")
    if isinstance(env.action_space, Box):
        assert env.action_space.shape == (
            1,
        ), "For Dyna you need a one dimension in the action space"
        env = DiscreteActionWrapper(env, dyna._n_actions)
    else:
        assert isinstance(
            env.action_space, Discrete
        ), "asserted Discrete action space in  given environment"
        logging.warning(
            f"Argument n_actions will be ignored. Instead use n_actions from given actions space: n={env.action_space.n}"
        )
    dyna.update_env(env)
    agent = dyna.get_agent(deterministic)

    done, truncated = False, False
    s, _ = env.reset()
    while not (done or truncated):
        a = agent.act(s)
        s, r, done, truncated, _ = env.step(a)
        env.render()

    print("truncated: ", truncated)
    print("done: ", done)
