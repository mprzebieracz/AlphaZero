from inference_service.inferer import InfererFactory
from mcts.core import MCTS


class MCTSFactory:
    def __init__(
        self,
        inferer_factory: InfererFactory,
        c_init=1.25,
        c_base=19652,
        eps=0.25,
        alpha=0.3,
    ) -> None:
        self._inferer_factory = inferer_factory
        self._c_init = c_init
        self._c_base = c_base
        self._eps = eps
        self._alpha = alpha

    def get_mcts(self) -> MCTS:
        return MCTS(
            self._inferer_factory.get_inferer(),
            self._c_init,
            self._c_base,
            self._eps,
            self._alpha,
        )
