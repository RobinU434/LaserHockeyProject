from pathlib import Path


def get_save_path(log_dir: Path, episode_idx: int, path: Path = None) -> Path:
    """creates save path

    Args:
        log_dir (Path): directory where to store checkpoint
        episode_idx (int): which episode
        path (Path, optional): overwrite. Defaults to None.

    Returns:
        Path: save path for checkpoint
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = log_dir / f"checkpoint_{episode_idx}.pt"
    return path
