class Entrypoint:
    """Entrypoint to experimentation landscape"""

    def __init__(self):
        """init object instance"""
        pass

    def train_dreamer(self):
        """_summary_"""
        pass

    def train_sb3_sac(self):
        """_summary_"""
        from project.scripts.train import train_sb3_sac

        train_sb3_sac()
