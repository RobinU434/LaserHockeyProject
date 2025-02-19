from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter


class _Logger(ABC):
    def __init__(self, log_dir: Path):
        super().__init__()

        self._log_dir = log_dir if isinstance(log_dir, Path) else Path(log_dir)

    def log_dict(self, d: Dict[str, float], step: int, prefix: str = ""):
        for key, value in d.items():
            self.log_scalar(prefix + key, value, step)

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError


class TensorBoardLogger(_Logger):
    def __init__(self, log_dir):
        super().__init__(log_dir)

        self._summary_writer = SummaryWriter(self._log_dir)

    def log_scalar(self, name, value, step):
        self._summary_writer.add_scalar(name, value, global_step=step)

    def save(self):
        pass


class CSVLogger(_Logger):
    def __init__(self, log_dir):
        super().__init__(log_dir)

        self._data = []

    def log_scalar(self, name, value, step):
        if not isinstance(value, (float, int)):
            logging.error(f"Value to name: {name} is not a float. Not able to log. ({value=}({type(value)}))")
            return
        self._data.append({name: value, "step": step})

    def save(self):
        df = pd.DataFrame(self._data)
        df = self._squeeze_dataframe(df, "step")
        df.to_csv(self._log_dir / "train_logs.csv")

    def _squeeze_dataframe(self, df: pd.DataFrame, index: str = "step"):
        return df.groupby(index).first().reset_index()
