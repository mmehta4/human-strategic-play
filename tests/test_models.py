import numpy as np
import pytest

from hsb.data.schema import Record
from hsb.eval import score
from hsb.games import Game, nash_equilibria, pure_nash, weakly_dominant_action
from hsb.generator import FAMILIES, suite
from hsb.models.structural import CognitiveHierarchy, LogitQRE, NashMix, QuantalLevelK, Uniform, logit_qre

PD = Game([[3, 0], [5, 1]], [[3, 0], [5, 1]])            # prisoner's dilemma, defect = action 1
MATCHING = Game([[1, -1], [-1, 1]], [[1, -1], [-1, 1]])  # own wants to match; set below


def rec(game, freq, pid="p", gid="g"):
    return Record("test", gid, pid, game, np.array(freq, float), 40.0)


def test_pure_nash_prisoners_dilemma():
    assert pure_nash(PD) == [(1, 1)]
    assert weakly_dominant_action(PD.own) == 1


def test_mixed_nash_matching_pennies():
    mp = Game([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    eqs = nash_equilibria(mp)
    assert len(eqs) == 1
    np.testing.assert_allclose(eqs[0][0], [0.5, 0.5], atol=1e-9)


def test_qre_limits():
    own, other = PD.own[None], PD.other[None]
    np.testing.assert_allclose(logit_qre(own, other, 0.0), [[0.5, 0.5]])
    assert logit_qre(own, other, 20.0)[0, 1] > 0.99  # converges to Nash (defect)


def test_from_2x2_layout():
    g = Game.from_2x2([21, 10, 24, 17], [2, 19, 6, 4])
    assert g.own[1, 0] == 24          # row plays 2nd action, col plays 1st
    assert g.other[0, 1] == 6         # col's payoff: col 1st action, row 2nd action
    assert g.swap().own[0, 1] == 6


def test_permutation_consistency():
    g = Game([[4, 1], [2, 3]], [[5, 0], [1, 2]])
    gp = g.permute([1, 0], [1, 0])
    m = QuantalLevelK()
    m.w, m.lam = np.array([0.2, 0.5, 0.3, 0.0]), 0.7
    p = m.predict([rec(g, [0.5, 0.5])])[0]
    pp = m.predict([rec(gp, [0.5, 0.5])])[0]
    np.testing.assert_allclose(p, pp[::-1], atol=1e-12)


@pytest.mark.parametrize("model", [Uniform(), NashMix(), LogitQRE(), QuantalLevelK(), CognitiveHierarchy()])
def test_models_fit_and_beat_or_match_uniform_in_sample(model):
    rng = np.random.default_rng(0)
    records = []
    for i in range(60):
        g = Game(rng.integers(1, 30, (2, 2)), rng.integers(1, 30, (2, 2)))
        # synthetic people: level-1 with noise
        u = g.own.mean(1)
        p = np.exp(0.3 * u) / np.exp(0.3 * u).sum()
        records.append(rec(g, p, f"p{i}", f"g{i}"))
    model.fit(records)
    pred = model.predict(records)
    assert np.allclose(pred.sum(1), 1)
    s = score(np.array([r.freq for r in records]), pred)
    assert s["nll"] <= np.log(2) + 1e-6


def test_quantal_level_k_recovers_level1_population():
    rng = np.random.default_rng(1)
    records = []
    for i in range(150):
        g = Game(rng.integers(1, 30, (2, 2)), rng.integers(1, 30, (2, 2)))
        u = g.own.mean(1)
        p = np.exp(0.4 * u) / np.exp(0.4 * u).sum()
        records.append(rec(g, p, f"p{i}", f"g{i}"))
    m = QuantalLevelK().fit(records)
    assert m.w[1] > 0.8 and abs(m.lam - 0.4) < 0.1


def test_generator_all_families():
    mechs = suite(n_per_family=3, seed=1)
    assert {m.family for m in mechs} == set(FAMILIES)
    for m in mechs:
        assert all(m.descriptions[r] for r in m.roles)
        if m.normal_form:
            g = m.game_for(m.roles[0])
            assert g.own.shape == (len(m.action_labels[m.roles[0]]), len(m.action_labels[m.roles[1]]))
            m.to_json()


def test_hybrid_v1_any_size_and_equivariant():
    torch = pytest.importorskip("torch")
    from hsb.models.neural import HybridQLkV1
    rng = np.random.default_rng(0)
    recs = []
    for i in range(40):  # mixed 2x2 and 3x3 games, synthetic level-1 players
        m = 2 if i % 2 else 3
        g = Game(rng.integers(1, 30, (m, m)), rng.integers(1, 30, (m, m)))
        u = g.own.mean(1); p = np.exp(0.3 * u) / np.exp(0.3 * u).sum()
        recs.append(rec(g, p, f"p{i}", f"g{i}"))
    model = HybridQLkV1(hidden=16, max_steps=200).fit(recs)
    preds = model.predict(recs)
    assert isinstance(preds, list) and {len(p) for p in preds} == {2, 3}
    g = recs[0].game
    perm = [2, 0, 1]
    p1 = model.predict([rec(g, [1/3] * 3)])[0]
    p2 = model.predict([rec(g.permute(perm, None), [1/3] * 3)])[0]
    np.testing.assert_allclose(p1[perm], p2, atol=1e-8)


def test_elementarynet_closer_valid_and_equivariant():
    pytest.importorskip("torch")
    from hsb.models.neural import ElementaryQCH
    rng = np.random.default_rng(0)
    recs = []
    for i in range(30):
        g = Game(rng.integers(1, 30, (3, 3)), rng.integers(1, 30, (3, 3)))
        u = g.own.mean(1); p = np.exp(0.3 * u) / np.exp(0.3 * u).sum()
        recs.append(rec(g, p, f"p{i}", f"g{i}"))
    model = ElementaryQCH(hidden=16, max_steps=100).fit(recs)
    preds = model.predict(recs)
    assert np.allclose(preds.sum(1), 1) and (preds > 0).all()
    g, perm = recs[0].game, [2, 0, 1]
    p1 = model.predict([rec(g, [1/3] * 3)])[0]
    p2 = model.predict([rec(g.permute(perm, None), [1/3] * 3)])[0]
    np.testing.assert_allclose(p1[perm], p2, atol=1e-8)


def test_free_llm_reply_parsing():
    from hsb.llm_free import _parse
    assert _parse('{"reasoning": "x", "percent": {"A": 30, "B": 70}}', 2) == {"A": 30.0, "B": 70.0}
    fenced = 'Sure!\n```json\n{"reasoning": "y", "percent": {"A": 55.5, "B": 44.5}}\n```'
    assert _parse(fenced, 2) == {"A": 55.5, "B": 44.5}
    assert _parse('no json here', 2) is None
    assert _parse('{"percent": {"A": 50}}', 2) is None           # missing an option


def test_nfg_parser_both_formats():
    from hsb.data.wlb import parse_nfg, standardise
    # payoff list: first player's action varies fastest, pairs (p1, p2)
    P = parse_nfg('NFG 1 R "t" { "1" "2" } { 2 2 }\n1 2 3 4 5 6 7 8')
    assert P[0, 0].tolist() == [1, 2] and P[1, 0].tolist() == [3, 4] and P[0, 1].tolist() == [5, 6]
    # outcome format
    Q = parse_nfg('NFG 1 R "t" { "1" "2" } { { "a" "b" } { "x" "y" } }\n""\n'
                  '{ { "o1" 1, 2 } { "o2" 3, 4 } }\n1 2 2 1')
    assert Q[0, 0].tolist() == [1, 2] and Q[1, 0].tolist() == [3, 4] and Q[1, 1].tolist() == [1, 2]
    S = standardise(P)
    assert np.allclose(S[:, :, 0].mean(), 0) and np.allclose(S[:, :, 0].std(ddof=1), 1)
