from abc import ABC, abstractmethod
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
import pandas as pd


class _Logger(ABC):
    def __init__(self, log_dir: Path):
        super().__init__()

        self._log_dir = log_dir

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError


class TensorBoardLogger(_Logger):
    def __init__(self, log_dir):
        super().__init__(log_dir)

        self._summary_writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, value, step):
        self._summary_writer.add_scalar(name, value, global_step=step)

    def save(self):
        pass

class CSVLogger(_Logger):
    def __init__(self, log_dir):
        super().__init__(log_dir)

        self._data = []

    def log_scalar(self, name, value, step):
        self._data.append({name: value, "step": step})

    def save(self):
        df = pd.DataFrame(self._data)
        df.to_csv(self._log_dir / "train_logs.csv")

        

