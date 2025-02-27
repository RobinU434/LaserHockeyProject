import sys

import logging
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


import gymnasium
from gymnasium.spaces import Box, Discrete

from project.algorithms.dyna.dyna import DynaQ
from project.algorithms.env_wrapper import Box2DiscreteActionWrapper
from project.algorithms.sac.sac import SAC
from project.environment.hockey_env.hockey.hockey_env import BasicOpponent
from project.environment.single_player_env import SinglePlayerHockeyEnv


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
        s, r, done, truncated, _ = env.step(a)
        env.render()


def render_sac_hockey(
    checkpoint: str | Path, deterministic: bool = False, strong_opponent: bool = False
):
    opponent = BasicOpponent(not strong_opponent)
    env = SinglePlayerHockeyEnv(opponent)
    sac = SAC.from_checkpoint(checkpoint, env)
    agent = sac.get_agent(deterministic)

    done, truncated = False, False
    s, _ = env.reset()
    while not (done or truncated):
        a = agent.act(s)
        s, r, done, truncated, _ = env.step(a)
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
        env = Box2DiscreteActionWrapper(env, dyna._n_actions)
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
