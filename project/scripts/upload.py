from logging import PlaceHolder
import numpy as np
import torch
from project.algorithms.env_wrapper import AffineActionTransform
from project.comprl.comprl.src.comprl.client import Agent, launch_client
from stable_baselines3.sac import SAC
from gymnasium import Env

class SBSACAgnet(Agent):
    def __init__(self, checkpoint: str):
        super().__init__()
        self.sac = SAC.load(checkpoint)
        self.action_transform = AffineActionTransform(
            None, np.array([1, 1, 1, 0.5]), np.array([0, 0, 0, 0.5])
        )

    def get_step(self, obv):
        if isinstance(obv, list):
            obv = np.array(obv)

        if isinstance(obv, np.ndarray):
            obv = torch.from_numpy(obv)

        batched = True          
        if len(obv.shape) == 1:
            batched = False
            obv = obv[None]

        mean_action, _, _ = self.sac.actor.get_action_dist_params(obv)
        mean_action = mean_action.detach().cpu().numpy()
        
        if not batched:
            mean_action = mean_action[0]

        mean_action = self.action_transform.action(mean_action)

        return mean_action.tolist()


def upload_sb_sac_agent(checkpoint, server_url, server_port, token):
    agent = SBSACAgnet(checkpoint)
    agent.run(token, server_url, server_port)
