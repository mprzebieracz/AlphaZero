from typing import Type, override

import torch
from inference_service.inferer import Inferer, InfererFactory
from network import AlphaZeroNetwork


class NetworkInferer(Inferer):
    def __init__(
        self,
        network_class: Type[AlphaZeroNetwork],
        network_file_path: str,
        device: torch.device,
    ):
        self._network = network_class.load_az_network(network_file_path, device)
        self._network.eval()
        self._network = torch.jit.script(self._network)

        self._device = next(self._network.parameters()).device

    @override
    def infer(self, game_state_tensor: torch.Tensor):
        return self._network(game_state_tensor)

    @property
    @override
    def device(self):
        return self._device


class InfererFactoryImpl(InfererFactory):
    def __init__(
        self,
        network_class: Type[AlphaZeroNetwork],
        network_file_path: str,
        device: torch.device,
    ):
        self._network_class = network_class
        self._network_file_path = network_file_path
        self._device = device

    @override
    def get_inferer(self):
        return NetworkInferer(
            self._network_class, self._network_file_path, self._device
        )
