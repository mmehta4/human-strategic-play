"""Arena game set (v1): 20 games in two weekly sets.

Honesty rule (no deception): opponents are real choices made by other players in the
same game and opposite role, and players are told exactly that. Until a game has
another player in the needed role, its result shows as "pending".

Every two-player game defines its action grid per role and a payoff rule, so the same
definition (1) scores players against each other and (2) turns arena data into normal-
form games for the benchmark (`normal_form`). Texts live in `i18n/*.json`, keyed by id.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Hand-made 2x2 games. MATRICES[g]["row"][i][j] / ["col"][i][j] = row's / col's payoff
# when row plays i and col plays j. Labels are neutral (A/B, X/Y) to avoid framing.
MATRICES = {
    "m_coord": {"row": [[30, 5], [10, 25]], "col": [[30, 10], [5, 25]]},        # two equilibria
    "m_other_dom": {"row": [[40, 10], [20, 30]], "col": [[10, 30], [5, 25]]},   # col has a dominant action
    "pd": {"row": [[30, 5], [40, 10]], "col": [[30, 40], [5, 10]]},             # prisoner's dilemma; 0 = cooperate
    "stag": {"row": [[40, 0], [30, 30]], "col": [[40, 30], [0, 30]]},           # stag hunt; 0 = stag
    "hawk_dove": {"row": [[0, 40], [10, 20]], "col": [[0, 10], [40, 20]]},      # chicken; 0 = hawk
    "hide_seek": {"row": [[0, 20], [20, 0]], "col": [[20, 0], [0, 20]]},        # row hides, col seeks
}
GAMBLE_GAINS = [10, 15, 20, 30]
SETS = ["core", "week2"]


@dataclass
class ArenaGame:
    id: str
    set: str
    kind: str                      # "number", "choice", "gambles"
    roles: list[str]
    min: float = 0
    max: float = 100
    step: float = 1
    opponent: dict = field(default_factory=dict)  # role -> opposite role (two-player games)
    group: int = 0                                # group size for n-player games

    def actions(self, role: str) -> list[float]:
        if self.kind == "choice":
            return [0.0, 1.0]
        n = int(round((self.max - self.min) / self.step))
        return [self.min + k * self.step for k in range(n + 1)]


def _pair(a, b):
    return {a: b, b: a}


GAMES = [
    # ---- core set (first visit)
    ArenaGame("beauty", "core", "number", ["player"], group=10),
    ArenaGame("ultimatum_offer", "core", "number", ["proposer"], 0, 100, 10, {"proposer": "responder"}),
    ArenaGame("ultimatum_min", "core", "number", ["responder"], 0, 100, 10, {"responder": "proposer"}),
    ArenaGame("trust_send", "core", "number", ["sender"], 0, 10, 1, {"sender": "receiver"}),
    ArenaGame("trust_return", "core", "number", ["receiver"], 0, 100, 10, {"receiver": "sender"}),
    ArenaGame("public_goods", "core", "number", ["member"], 0, 20, 1, group=4),
    ArenaGame("auction", "core", "number", ["bidder"], 0, 100, 1),
    ArenaGame("m_coord", "core", "choice", ["row", "col"], opponent=_pair("row", "col")),
    ArenaGame("m_other_dom", "core", "choice", ["row", "col"], opponent=_pair("row", "col")),
    ArenaGame("gambles", "core", "gambles", ["player"]),
    # ---- week-two set (second visit)
    ArenaGame("dictator", "week2", "number", ["giver"], 0, 100, 10),
    ArenaGame("pd", "week2", "choice", ["row", "col"], opponent=_pair("row", "col")),
    ArenaGame("stag", "week2", "choice", ["row", "col"], opponent=_pair("row", "col")),
    ArenaGame("hawk_dove", "week2", "choice", ["row", "col"], opponent=_pair("row", "col")),
    ArenaGame("hide_seek", "week2", "choice", ["row", "col"], opponent=_pair("row", "col")),
    ArenaGame("second_price", "week2", "number", ["bidder"], 0, 100, 1),
    ArenaGame("bertrand", "week2", "number", ["seller"], 1, 10, 1, {"seller": "seller"}),
    ArenaGame("weakest_link", "week2", "number", ["member"], 1, 7, 1, group=4),
    ArenaGame("money_request", "week2", "number", ["player"], 11, 20, 1, {"player": "player"}),
    ArenaGame("travelers", "week2", "number", ["claimant"], 20, 100, 10, {"claimant": "claimant"}),
]
BY_ID = {g.id: g for g in GAMES}


def games_in_set(set_id: str) -> list[ArenaGame]:
    return [g for g in GAMES if g.set == set_id]


def set_for_session(session_no: int) -> str:
    """Session 1 plays the core set, session 2 the week-two set, then they alternate,
    so every replay shows the player how their choices changed."""
    return SETS[(session_no - 1) % len(SETS)]


# ------------------------------------------------------------------ two-player payoffs

def payoff(game_id: str, role: str, mine: float, theirs: float) -> float:
    """Points for `role` choosing `mine` against an opponent choosing `theirs`."""
    if game_id in MATRICES:
        m = MATRICES[game_id]
        return m["row"][int(mine)][int(theirs)] if role == "row" else m["col"][int(theirs)][int(mine)]
    if game_id == "ultimatum_offer":       # mine = offer, theirs = responder's minimum
        return 100 - mine if mine >= theirs else 0
    if game_id == "ultimatum_min":         # mine = minimum, theirs = offer
        return theirs if theirs >= mine else 0
    if game_id == "trust_send":            # mine = points sent, theirs = % returned
        return round(10 - mine + theirs / 100 * 3 * mine, 1)
    if game_id == "trust_return":          # mine = % returned, theirs = points sent
        return round((1 - mine / 100) * 3 * theirs, 1)
    if game_id == "bertrand":
        return 10 * mine if mine < theirs else (5 * mine if mine == theirs else 0)
    if game_id == "money_request":
        return mine + (20 if mine == theirs - 1 else 0)
    if game_id == "travelers":
        if mine == theirs:
            return mine
        return mine + 10 if mine < theirs else theirs - 10
    raise KeyError(game_id)


def normal_form(game_id: str, role: str) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    """(my actions, their actions, my payoff matrix, their payoff matrix as (their, mine))
    for a two-player arena game, ready to become an `hsb.games.Game`."""
    g = BY_ID[game_id]
    other_role = g.opponent[role]
    other_game = counterpart(game_id)
    mine_a, theirs_a = g.actions(role), BY_ID[other_game].actions(other_role)
    own = [[payoff(game_id, role, a, b) for b in theirs_a] for a in mine_a]
    other = [[payoff(other_game, other_role, b, a) for a in mine_a] for b in theirs_a]
    return mine_a, theirs_a, own, other


def counterpart(game_id: str) -> str:
    """The arena game the opponent played (ultimatum and trust split roles across two screens)."""
    return {"ultimatum_offer": "ultimatum_min", "ultimatum_min": "ultimatum_offer",
            "trust_send": "trust_return", "trust_return": "trust_send"}.get(game_id, game_id)


def matrix_lines(game_id: str, role: str, t) -> list[str]:
    m = MATRICES[game_id]
    mine = m["row"] if role == "row" else _transpose(m["col"])
    theirs = m["col"] if role == "row" else _transpose(m["row"])
    my_labels, their_labels = (["A", "B"], ["X", "Y"]) if role == "row" else (["X", "Y"], ["A", "B"])
    lines = [t("matrix.intro")]
    for i, a in enumerate(my_labels):
        for j, b in enumerate(their_labels):
            lines.append(t("matrix.cell", mine=a, theirs=b, me=mine[i][j], them=theirs[i][j]))
    return lines


def option_labels(role: str) -> list[str]:
    return ["A", "B"] if role == "row" else ["X", "Y"]


def _transpose(x):
    return [list(r) for r in zip(*x)]
