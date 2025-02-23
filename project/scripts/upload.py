from project.algorithms.sb3_extensions.sac import SB_SAC_Agent


def upload_sb_sac_agent(checkpoint, server_url, server_port, token):
    agent = SB_SAC_Agent.from_checkpoint(checkpoint)
    agent.run(token, server_url, server_port)
