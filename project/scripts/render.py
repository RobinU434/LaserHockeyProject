from pathlib import Path

import gymnasium

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
            render_mode ="human", 
        )

    sac = SAC.from_checkpoint(checkpoint, env)
    agent = sac.get_agent(deterministic)

    done, truncated = False, False
    s, _ = env.reset()
    while not (done or truncated):
        a = agent.act(s)
        s, r, truncated, done, _ = env.step(a)
        env.render()
