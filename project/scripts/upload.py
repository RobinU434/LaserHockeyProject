import numpy as np
import torch
from project.comprl.comprl.src.comprl.client import Agent, launch_client
from stable_baselines3.sac import SAC


class SBSACAgnet(Agent):
    def __init__(self, checkpoint: str):
        super().__init__()
        self.sac = SAC.load(checkpoint)

    def get_step(self, obv):
        if isinstance(obv, list):
            obv = np.array(obv)

        if isinstance(obv, np.ndarray):
            obv = torch.from_numpy(obv)

        mean_action, _, _ = self.sac.actor.get_action_dist_params(obv)
        mean_action = mean_action.detach().cpu().numpy().tolist()
        return mean_action


def upload_sb_sac_agent(checkpoint, server_url, server_port, token):
    agent = SBSACAgnet(checkpoint)
    agent.run(token, server_url, server_port)
