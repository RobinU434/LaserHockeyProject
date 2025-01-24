from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, sate: np.ndarray) -> np.ndarray:
        """receives a state and returns an action

        Args:
            sate (np.ndarray): state representation directly from the environment

        Returns:
            np.ndarray: action which fits in the action space of the environment
        """
        raise NotImplementedError