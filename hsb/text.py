"""Plain-language descriptions of games, used as LLM inputs and in the arena.
Describing every mechanism in words is what lets one model read any rule."""
from __future__ import annotations

from .games import Game

ROW_LABELS = ["A", "B", "C", "D", "E", "F"]
COL_LABELS = ["X", "Y", "Z", "W", "V", "U"]


def describe_matrix_game(game: Game, currency: str = "points") -> str:
    m, n = game.own.shape
    lines = [
        f"You and another participant each choose an option at the same time, without "
        f"seeing each other's choice. You choose between {', '.join(ROW_LABELS[:m])}; "
        f"the other participant chooses between {', '.join(COL_LABELS[:n])}.",
        "Your earnings and theirs depend on both choices:",
    ]
    for i in range(m):
        for j in range(n):
            lines.append(f"- You choose {ROW_LABELS[i]} and they choose {COL_LABELS[j]}: "
                         f"you get {game.own[i, j]:g} {currency}, they get {game.other[j, i]:g} {currency}.")
    lines.append("You play this game once, with a stranger, and never meet them again.")
    return "\n".join(lines)
