import numpy as np
import torch
from project.algorithms.dyna.dyna import DynaQ
from project.algorithms.sb3_extensions.sac import SBSACCompAgent
from project.algorithms.utils.gym_helper import ContinuousPlaceHolderEnv
from project.comprl.comprl.src.comprl.client import Agent
from project.algorithms.env_wrapper import (
    Box2MultiDiscreteActionWrapper,
    MD2DiscreteActionWrapper,
)


class DynaCompAgent(Agent):
    def __init__(self, dyna: DynaQ):
        super().__init__()
        self.dyna = dyna
        self.place_holder_env = ContinuousPlaceHolderEnv(18, 4, 2, 0)
        self.box2md = Box2MultiDiscreteActionWrapper(
            self.place_holder_env, np.array([10, 10, 10, 2])
        )
        self.md2d = MD2DiscreteActionWrapper(self.box2md)

    @classmethod
    def from_checkpoint(cls, checkpoint: str) -> "DynaCompAgent":
        return cls(DynaQ.from_checkpoint(checkpoint))

    def get_step(self, obv):
        obv = torch.tensor(obv)
        q_values = self.dyna.q_net.complete_forward(obv)
        d_action = torch.argmax(q_values)
        md_action = self.md2d.action(d_action)
        c_action = self.box2md.action(md_action)
        return c_action.tolist()


def upload_sb_sac_agent(checkpoint, server_url, server_port, token):
    agent = SBSACCompAgent.from_checkpoint(checkpoint)
    agent.run(token, server_url, server_port)


def upload_dyna_agent(checkpoint, server_url, server_port, token):
    agent = DynaCompAgent.from_checkpoint(checkpoint)
    agent.run(token, server_url, server_port)
