"""Procedural mechanism generator.

Generates an unlimited supply of strategic situations across families, each with
(1) a formal spec, (2) a plain-language description per role, and (3) where it is a
finite two-player game, its normal form, so every structural baseline can score it.
Training across many generated families, and testing on families held out, is how
the model is pushed to learn structure rather than wording.

Families: matrix, beauty_contest, ultimatum, trust, public_goods,
          first_price_auction, second_price_auction, bertrand
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .games import Game
from .text import describe_matrix_game


@dataclass
class Mechanism:
    family: str
    params: dict
    roles: list[str]
    descriptions: dict[str, str]
    action_labels: dict[str, list[str]]
    normal_form: dict | None = None   # {"role": [[own payoffs]], ...} for 2-player finite games
    reference: dict = field(default_factory=dict)  # textbook benchmarks (equilibrium, level-k)

    def game_for(self, role: str) -> Game | None:
        if not self.normal_form:
            return None
        r0, r1 = self.roles
        own = np.array(self.normal_form[role])
        other = np.array(self.normal_form[r1 if role == r0 else r0])
        return Game(own, other, {"family": self.family})

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _grid(lo, hi, step):
    return [round(x, 6) for x in np.arange(lo, hi + 1e-9, step)]


# ----------------------------------------------------------------------------- families

def matrix(rng, m=2, n=2, lo=1, hi=50) -> Mechanism:
    own = rng.integers(lo, hi + 1, size=(m, n))
    other = rng.integers(lo, hi + 1, size=(n, m))
    g = Game(own, other)
    return Mechanism(
        "matrix", {"m": m, "n": n, "payoff_range": [lo, hi]}, ["row", "col"],
        {"row": describe_matrix_game(g), "col": describe_matrix_game(g.swap())},
        {"row": ["A", "B", "C", "D"][:m], "col": ["A", "B", "C", "D"][:n]},
        {"row": own.tolist(), "col": other.tolist()})


def beauty_contest(rng) -> Mechanism:
    p = float(rng.choice([1 / 2, 2 / 3, 3 / 4, 4 / 5]))
    hi = int(rng.choice([100, 200]))
    n = int(rng.choice([2, 3, 5, 10, 20]))
    prize = int(rng.choice([10, 20, 50]))
    frac = {0.5: "one half", 2 / 3: "two thirds", 0.75: "three quarters", 0.8: "four fifths"}[round(p, 4) if p != 2 / 3 else 2 / 3]
    text = (f"You and {n - 1} other participant{'s' if n > 2 else ''} each pick a whole number from 0 to {hi}. "
            f"The person whose number is closest to {frac} of the average of all numbers picked wins "
            f"{prize} points; ties split the prize. Everyone else gets nothing.")
    level0 = hi / 2
    return Mechanism("beauty_contest", {"p": p, "max": hi, "n_players": n, "prize": prize},
                     ["player"], {"player": text}, {"player": [str(x) for x in range(hi + 1)]},
                     reference={"nash": 0, "level_k_guess": {k: round(level0 * p ** k, 2) for k in range(5)}})


def ultimatum(rng) -> Mechanism:
    pie = int(rng.choice([10, 20, 100]))
    step = pie // 10
    offers = _grid(0, pie, step)
    # strategy method: the responder states the smallest offer they accept
    prop = [[(pie - o) if o >= t else 0 for t in offers] for o in offers]
    resp = [[o if o >= t else 0 for o in offers] for t in offers]
    return Mechanism(
        "ultimatum", {"pie": pie, "step": step}, ["proposer", "responder"],
        {"proposer": (f"You have {pie} points to split with another participant. You offer them part of it. "
                      f"If they accept, you keep the rest and they get your offer. If they reject, you both "
                      f"get nothing. How much do you offer (0 to {pie}, in steps of {step})?"),
         "responder": (f"Another participant has {pie} points to split with you and will offer you part of "
                       f"it. If you accept, you get the offer and they keep the rest; if you reject, you both get "
                       f"nothing. What is the smallest offer you would accept (0 to {pie}, in steps of {step})?")},
        {"proposer": [str(o) for o in offers], "responder": [str(t) for t in offers]},
        {"proposer": prop, "responder": resp},
        reference={"subgame_perfect": {"offer": step, "min_accept": 0}})


def trust(rng) -> Mechanism:
    e = int(rng.choice([10, 20]))
    k = int(rng.choice([2, 3, 4]))
    sends = _grid(0, e, e // 5)
    shares = _grid(0, 0.8, 0.2)  # share of the received amount the trustee returns
    send_pay = [[e - s + r * k * s for r in shares] for s in sends]
    ret_pay = [[(1 - r) * k * s for s in sends] for r in shares]
    return Mechanism(
        "trust", {"endowment": e, "multiplier": k}, ["sender", "receiver"],
        {"sender": (f"You have {e} points. You can send any part of it to another participant; whatever you "
                    f"send is multiplied by {k}. They then decide how much of the multiplied amount to send "
                    f"back to you. How much do you send?"),
         "receiver": (f"Another participant has {e} points and may send you part of it; you receive {k} times "
                      f"what they send. What share of what you receive will you send back to them?")},
        {"sender": [str(s) for s in sends], "receiver": [f"{int(r * 100)}%" for r in shares]},
        {"sender": send_pay, "receiver": ret_pay},
        reference={"subgame_perfect": {"send": 0, "return_share": 0}})


def public_goods(rng) -> Mechanism:
    e = int(rng.choice([10, 20]))
    n = int(rng.choice([2, 3, 4]))
    mpcr = float(rng.choice([0.4, 0.5, 0.6, 0.75])) if n > 2 else float(rng.choice([0.6, 0.75, 0.9]))
    contribs = _grid(0, e, e // 5)
    text = (f"You are in a group of {n}. Each of you has {e} points and decides how many to put into a group "
            f"project. Every point put in by anyone pays each group member {mpcr:g} points. You keep the points you "
            f"don't contribute. How many points do you contribute?")
    mech = Mechanism("public_goods", {"endowment": e, "group_size": n, "mpcr": mpcr}, ["member"],
                     {"member": text}, {"member": [str(c) for c in contribs]},
                     reference={"nash_contribution": 0 if mpcr < 1 else e, "social_optimum": e if mpcr * n > 1 else 0})
    if n == 2:
        pay = [[e - c + mpcr * (c + d) for d in contribs] for c in contribs]
        mech.roles = ["member_a", "member_b"]
        mech.descriptions = {"member_a": text, "member_b": text}
        mech.action_labels = {"member_a": mech.action_labels["member"], "member_b": mech.action_labels["member"]}
        mech.normal_form = {"member_a": pay, "member_b": pay}
    return mech


def first_price_auction(rng) -> Mechanism:
    n = int(rng.choice([2, 3, 5]))
    vmax = int(rng.choice([50, 100]))
    value = int(rng.integers(1, vmax + 1))
    text = (f"You are bidding for an item in a sealed-bid auction against {n - 1} other bidder"
            f"{'s' if n > 2 else ''}. The item is worth {value} points to you. Each other bidder's value is a "
            f"random whole number from 1 to {vmax}, and they know only their own value. The highest bid wins and "
            f"pays its own bid; you earn your value minus your bid if you win, and nothing otherwise. What do you bid?")
    return Mechanism("first_price_auction", {"n_bidders": n, "value_max": vmax, "value": value}, ["bidder"],
                     {"bidder": text}, {"bidder": [str(b) for b in range(vmax + 1)]},
                     reference={"risk_neutral_bne_bid": round(value * (n - 1) / n, 2)})


def second_price_auction(rng) -> Mechanism:
    m = first_price_auction(rng)
    m.family = "second_price_auction"
    m.descriptions["bidder"] = m.descriptions["bidder"].replace(
        "The highest bid wins and pays its own bid", "The highest bid wins but pays the second-highest bid")
    m.descriptions["bidder"] = m.descriptions["bidder"].replace(
        "you earn your value minus your bid if you win",
        "you earn your value minus the second-highest bid if you win")
    m.reference = {"dominant_bid": m.params["value"]}
    return m


def bertrand(rng) -> Mechanism:
    cost = int(rng.choice([0, 2, 4]))
    prices = _grid(cost + 1, cost + 10, 1)
    demand = int(rng.choice([10, 20]))
    def pay(p, q):
        if p < q:
            return (p - cost) * demand
        if p == q:
            return (p - cost) * demand / 2
        return 0
    table = [[pay(p, q) for q in prices] for p in prices]
    text = (f"You and one competing seller sell an identical product to {demand} buyers. Each unit costs you "
            f"{cost} points to make. You each set a price from {prices[0]} to {prices[-1]} at the same time. All "
            f"buyers buy from the cheaper seller; if prices are equal they split evenly. What price do you set?")
    return Mechanism("bertrand", {"cost": cost, "demand": demand}, ["seller_a", "seller_b"],
                     {"seller_a": text, "seller_b": text},
                     {"seller_a": [str(p) for p in prices], "seller_b": [str(p) for p in prices]},
                     {"seller_a": table, "seller_b": table},
                     reference={"nash_price": [cost + 1, cost + 2]})


FAMILIES = {
    "matrix": matrix, "beauty_contest": beauty_contest, "ultimatum": ultimatum, "trust": trust,
    "public_goods": public_goods, "first_price_auction": first_price_auction,
    "second_price_auction": second_price_auction, "bertrand": bertrand,
}


def generate(family: str, seed: int | None = None, **kw) -> Mechanism:
    return FAMILIES[family](np.random.default_rng(seed), **kw)


def suite(n_per_family: int = 50, seed: int = 0) -> list[Mechanism]:
    rng = np.random.default_rng(seed)
    return [FAMILIES[f](rng) for f in FAMILIES for _ in range(n_per_family)]
