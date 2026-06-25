"""Connect4 perfect-play engine (Cumulative negamax, C++ backend)."""

from __future__ import annotations

import sys

sys.path.append("../build/engine/")
from engine_bind import Connect4PerfectEngine as _CppEngine  # pyright: ignore


class PerfectAgent:
    def __init__(self) -> None:
        self._engine = _CppEngine()

    def act(self, game_state) -> int:
        col = self._engine.best_move(
            game_state.get_board_state(),
            game_state.current_player,
        )
        self._engine.clear_cache()
        return col
