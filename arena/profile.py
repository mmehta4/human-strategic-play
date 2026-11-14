"""Turn a player's choices into a strategic profile, results against other real players,
percentiles against everyone, and how they changed between visits. Measured from
behaviour, never self-report."""
from __future__ import annotations

import math
import random

from .games import BY_ID, GAMBLE_GAINS, counterpart, payoff


def reasoning_depth(guess: float) -> float:
    """Level-k reading of a 2/3-beauty-contest guess: level 0 guesses 50 on average,
    level k guesses 50 * (2/3)^k."""
    if guess >= 50:
        return 0.0
    if guess <= 0.5:
        return 5.0  # at or near the equilibrium of 0
    return round(min(5.0, math.log(guess / 50) / math.log(2 / 3)), 1)


def loss_aversion(accepted: list[bool]) -> float | None:
    """Lambda from 50/50 gambles (win G, lose 10): the smallest accepted gain over 10."""
    gains = [g for g, a in zip(GAMBLE_GAINS, accepted) if a]
    return min(gains) / 10 if gains else None  # None: rejects even win 30 / lose 10


# key -> (kind, top) ; kind: "steps" | "pct" | "x" | "yes" (binary) | "of" (value out of top)
TRAITS = {
    "reasoning_depth": ("steps", 5), "depth_1120": ("steps", 9),
    "fair_offer_pct": ("pct", 100), "min_accept_pct": ("pct", 100), "dictator_pct": ("pct", 100),
    "trust_pct": ("pct", 100), "reciprocity_pct": ("pct", 100), "cooperation_pct": ("pct", 100),
    "bid_shading_pct": ("pct", 100), "truthful_bid_pct": ("pct", 100),
    "loss_aversion": ("x", 3),
    "pd_cooperate": ("yes", 1), "stag": ("yes", 1), "hawk": ("yes", 1),
    "price": ("of", 10), "effort": ("of", 7), "claim": ("of", 100),
}


def traits(choices: dict) -> dict:
    """choices: game_id -> {"choice": ..., "role": ..., "context": {...}}"""
    c = {k: v["choice"] for k, v in choices.items()}
    ctx = {k: v.get("context") or {} for k, v in choices.items()}
    t = {}
    if "beauty" in c:
        t["reasoning_depth"] = reasoning_depth(c["beauty"])
    if "money_request" in c:
        t["depth_1120"] = 20 - c["money_request"]  # level 0 asks for 20; each step down is one more step
    if "ultimatum_offer" in c:
        t["fair_offer_pct"] = c["ultimatum_offer"]
    if "ultimatum_min" in c:
        t["min_accept_pct"] = c["ultimatum_min"]
    if "dictator" in c:
        t["dictator_pct"] = c["dictator"]
    if "trust_send" in c:
        t["trust_pct"] = c["trust_send"] * 10
    if "trust_return" in c:
        t["reciprocity_pct"] = c["trust_return"]
    if "public_goods" in c:
        t["cooperation_pct"] = c["public_goods"] * 5
    if "auction" in c and ctx["auction"].get("value"):
        t["bid_shading_pct"] = round(100 * (1 - c["auction"] / ctx["auction"]["value"]), 1)
    if "second_price" in c and ctx["second_price"].get("value"):
        v = ctx["second_price"]["value"]
        t["truthful_bid_pct"] = round(max(0.0, 100 - 100 * abs(c["second_price"] - v) / v), 1)
    if "gambles" in c:
        t["loss_aversion"] = loss_aversion(ctx["gambles"]["accepted"])
    if "pd" in c:
        t["pd_cooperate"] = int(c["pd"] == 0)
    if "stag" in c:
        t["stag"] = int(c["stag"] == 0)
    if "hawk_dove" in c:
        t["hawk"] = int(c["hawk_dove"] == 0)
    if "bertrand" in c:
        t["price"] = c["bertrand"]
    if "weakest_link" in c:
        t["effort"] = c["weakest_link"]
    if "travelers" in c:
        t["claim"] = c["travelers"]
    return t


def results(pid: str, choices: dict, pool: dict) -> dict:
    """Points in each game against other real players' choices. `pool[game_id][role]` is a
    list of (participant_id, choice, context) from other unflagged players. Opponents are
    drawn with a seed tied to this player, so results are stable for a given pool."""
    rng = random.Random(pid)
    out = {}

    def others(game, role, k=1):
        cands = sorted((c for c in pool.get(game, {}).get(role, []) if c[0] != pid), key=lambda c: c[0])
        return rng.sample(cands, k) if len(cands) >= k else None

    for gid, mine in choices.items():
        g = BY_ID.get(gid)
        if g is None:
            continue
        x, role = mine["choice"], mine["role"]
        if g.opponent:  # standard two-player game
            if o := others(counterpart(gid), g.opponent[role]):
                out[gid] = payoff(gid, role, x, o[0][1])
        elif gid == "beauty":
            pool_b = sorted((c for c in pool.get("beauty", {}).get("player", []) if c[0] != pid), key=lambda c: c[0])
            if pool_b:
                group = rng.sample(pool_b, min(9, len(pool_b)))
                guesses = [x] + [o[1] for o in group]
                target = 2 / 3 * sum(guesses) / len(guesses)
                dist = [abs(v - target) for v in guesses]
                winners = [i for i, d in enumerate(dist) if d == min(dist)]
                out[gid] = round(100 / len(winners), 1) if 0 in winners else 0
        elif gid == "public_goods":
            if o := others(gid, "member", 3):
                out[gid] = round(20 - x + 0.5 * (x + sum(v[1] for v in o)), 1)
        elif gid == "weakest_link":
            if o := others(gid, "member", 3):
                out[gid] = 60 + 20 * min([x] + [v[1] for v in o]) - 10 * x
        elif gid in ("auction", "second_price"):
            if o := others(gid, "bidder"):
                value, their_bid = mine["context"]["value"], o[0][1]
                win = x > their_bid or (x == their_bid and rng.random() < 0.5)
                price = x if gid == "auction" else their_bid
                out[gid] = value - price if win else 0
        elif gid == "dictator":
            out[gid] = 100 - x
        elif gid == "gambles":
            flips = [rng.random() < 0.5 for _ in GAMBLE_GAINS]
            acc = mine["context"]["accepted"]
            out[gid] = sum((gain if heads else -10) for gain, a, heads in zip(GAMBLE_GAINS, acc, flips) if a)
    return out


def percentile(value, population: list) -> int | None:
    vals = [v for v in population if v is not None]
    if value is None or len(vals) < 5:
        return None
    return round(100 * sum(v < value for v in vals) / len(vals))


def share_same(value, population: list) -> int | None:
    """For yes/no traits: % of other players who made the same choice."""
    vals = [v for v in population if v is not None]
    if value is None or len(vals) < 5:
        return None
    return round(100 * sum(v == value for v in vals) / len(vals))


def changes(first: dict, latest: dict) -> dict:
    """Trait differences between a player's first and latest play of the same games."""
    a, b = traits(first), traits(latest)
    return {k: round(b[k] - a[k], 1) for k in a if k in b and a[k] is not None and b[k] is not None and b[k] != a[k]}
