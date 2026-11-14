"""The 12-study lab and online dataset used by GameNet and ElementaryNet ("all12"): 366 games
(mostly 3x3, some larger) from Stahl & Wilson (1994, 1995), Costa-Gomes et al. (1998),
Goeree & Holt (2001), Cooper & Van Huyck (2003), Haruvy et al. (2001), Haruvy & Stahl (2007),
Stahl & Haruvy (2008), Rogers et al. (2009), Costa-Gomes et al. (2008), Fudenberg & Liang
(2019) and Chui, Hartline & Wright (2023).

The files come from github.com/gregdeon/elementarynet at a pinned commit (fetched by
`scripts/fetch_wlb_data.sh`; that repository has no license, so nothing is copied here). This
module only *reads* their data files: .nfg game files, the choice counts written in their
datasets.py (parsed as a literal, never executed), a .mat file and a numpy-only pickle. None of
their code runs.

Payoffs are standardised per game and per player (mean 0, s.d. 1), as in their main
experiments, and identical games are merged by summing choice counts.
"""
from __future__ import annotations

import ast
import io
import os
import pickle
import re
import subprocess
from pathlib import Path

import numpy as np

from ..games import Game
from .schema import Record

COMMIT = "69005d68b367b8cab4a51327ae00061d2c545d90"
DEFAULT_DIR = Path.home() / ".cache" / "hsb" / "elementarynet"
STUDIES = ["sw94", "sw95", "cgcb98", "gh01", "cvh03", "hsw01", "hs07", "sh08", "rpc08", "cgw08", "fl19", "chw23"]


def repo_dir() -> Path:
    path = Path(os.environ.get("HSB_WLB_DIR", DEFAULT_DIR))
    if not (path / "torch_bgt" / "data").exists():
        raise FileNotFoundError(f"data not found at {path}; run scripts/fetch_wlb_data.sh")
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        raise RuntimeError(f"data repo is at {head}, expected pinned commit {COMMIT}")
    return path


def _study_table(path: Path) -> dict:
    """The `datasets = {...}` literal from their datasets.py, with named constants set to None."""
    tree = ast.parse((path / "torch_bgt" / "datasets.py").read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "datasets" for t in n.targets))

    class NamesToNone(ast.NodeTransformer):
        def visit_Name(self, n):
            return ast.copy_location(ast.Constant(None), n)

    node = ast.fix_missing_locations(NamesToNone().visit(node))
    return ast.literal_eval(node)


def _tokens(text):
    return re.findall(r'\{|\}|"[^"]*"|[^\s{}]+', text)


def _nest(tokens):
    stack = [[]]
    for t in tokens:
        if t == "{":
            stack.append([])
        elif t == "}":
            inner = stack.pop()
            stack[-1].append(inner)
        else:
            stack[-1].append(t)
    return stack[0]


def parse_nfg(text: str) -> np.ndarray:
    """Gambit .nfg (version 1, 2 players) -> payoffs (rows, cols, 2). The first player's
    action varies fastest in the payoff list, as in the Gambit format."""
    parsed = _nest(_tokens(text))
    while not isinstance(parsed[0], list):
        parsed.pop(0)
    action_list, rest = parsed[1], parsed[2:]
    if rest and rest[0] == '""':
        rest = rest[1:]
    m, n = [int(a) if isinstance(a, str) else len(a) for a in action_list]
    P = np.zeros((m, n, 2))
    if isinstance(rest[0], list):  # outcome-based format
        outcomes = np.zeros((len(rest[0]) + 1, 2))
        for i, o in enumerate(rest[0]):
            outcomes[i + 1] = [float(o[k + 1].strip(",")) for k in range(2)]
        idx = list(map(int, rest[1:]))
        for k, (c, r) in enumerate((c, r) for c in range(n) for r in range(m)):
            P[r, c] = outcomes[idx[k]]
    else:
        vals = list(map(float, rest))
        k = 0
        for c in range(n):
            for r in range(m):
                P[r, c] = vals[k:k + 2]
                k += 2
    return P


class _NumpyOnly(pickle.Unpickler):
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module, name)


def _raw_games(path: Path) -> list[tuple[str, str, np.ndarray, list[np.ndarray]]]:
    """(study, game_id, payoffs, [row counts, col counts]) for every game in all12, unmerged."""
    data = path / "torch_bgt" / "data"
    table = _study_table(path)
    out = []
    for study in STUDIES:
        for entry in table[study]:
            mode, rest = entry[1], entry[2:]
            if mode == "wlb":
                fname, counts = rest[0], rest[1]
                row_perm = rest[2] if len(rest) > 2 else None
                col_perm = rest[3] if len(rest) > 3 else None
                P = parse_nfg((data / fname).read_text())
                if row_perm is not None:
                    P = P[row_perm]
                if col_perm is not None:
                    P = P[:, col_perm]
                out.append((study, fname, P, [np.array(c, float) for c in counts]))
            elif mode == "fl":
                from scipy.io import loadmat
                mat = loadmat(data / rest[0])
                for i in range(len(mat["freq"])):
                    P = np.stack([mat["row"][i][0], mat["column"][i][0]], axis=2).astype(float)
                    out.append((study, f"{rest[1]}_{i + 1}", P, [mat["freq"][i, 0:3].astype(float), np.zeros(3)]))
            elif mode == "chw":
                obs = _NumpyOnly(io.BytesIO((data / rest[0]).read_bytes())).load()
                counts = np.asarray(obs).sum(axis=0)
                for i in range(len(counts)):
                    P = parse_nfg((data / f"{rest[1]}_{i}.nfg").read_text())
                    out.append((study, f"{rest[2]}_{i}", P, [counts[i].astype(float), np.zeros(3)]))
    return out


def standardise(P: np.ndarray) -> np.ndarray:
    """Per player: mean 0, s.d. 1 (sample s.d., as torch.std). Constant payoffs are only centred."""
    out = np.empty_like(P)
    for k in range(2):
        x = P[:, :, k]
        sd = x.std(ddof=1)
        out[:, :, k] = (x - x.mean()) / (sd if sd > 0 else 1.0)
    return out


def load(path: Path | None = None) -> list[Record]:
    path = Path(path) if path else repo_dir()
    merged: list[list] = []  # [study, game_id, payoffs, counts]
    for study, gid, P, counts in _raw_games(path):
        P = standardise(P)
        for g in merged:
            if g[2].shape == P.shape and P.shape[0] <= 10 and np.allclose(g[2], P):
                g[1] += f"; {gid}"
                g[3] = [a + b for a, b in zip(g[3], counts)]
                break
        else:
            merged.append([study, gid, P, counts])
    records = []
    for study, gid, P, counts in merged:
        views = [(P[:, :, 0], P[:, :, 1].T), (P[:, :, 1].T, P[:, :, 0])]
        for role, (own, other), c in zip(("row", "col"), views, counts):
            if c.sum() > 0:
                records.append(Record(source="wlb", game_id=f"wlb-{gid}", problem_id=f"wlb-{gid}-{role}",
                                      game=Game(own, other), freq=c / c.sum(), n=float(c.sum()), family=study))
    return records


def standardise_records(records: list[Record]) -> list[Record]:
    """Apply the same per-game, per-player standardisation to any records (e.g. games2p2k), so
    models trained on one dataset see payoffs on the same scale as the test dataset."""
    out = []
    for r in records:
        P = standardise(np.stack([r.game.own, r.game.other.T], axis=2))
        out.append(Record(r.source, r.game_id, r.problem_id, Game(P[:, :, 0], P[:, :, 1].T, meta=r.game.meta),
                          r.freq, r.n, r.family, r.population, r.extra))
    return out
