from abc import ABC, abstractmethod
from typing import Any, Tuple
import torch


class Inferer(ABC):
    @abstractmethod
    def infer(self, game_state_tensor: torch.Tensor) -> Tuple[Any, Any]:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass


class InfererFactory(ABC):
    @abstractmethod
    def get_inferer(self) -> Inferer:
        pass
