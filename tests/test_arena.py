import json
import random
import re
import string

import pytest

from arena.games import BY_ID, GAMBLE_GAINS, GAMES, MATRICES, games_in_set, normal_form, payoff
from arena.i18n import LANGS, STRINGS, missing_keys


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_ADMIN_TOKEN", "test-token")
    import arena.app as appmod
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "arena.db")
    appmod.app.config["TESTING"] = True
    return appmod.app


def csrf(html):
    return re.search(r'name="csrf" value="([0-9a-f]+)"', html).group(1)


def answer(c, rng, path):
    page = c.get(path).get_data(as_text=True)
    data = {"csrf": csrf(page), "rt_ms": rng.randint(1500, 9000)}
    if 'name="g0"' in page:
        data.update({f"g{i}": rng.choice(["play", "pass"]) for i in range(len(GAMBLE_GAINS))})
    elif 'type="number"' in page:
        lo, hi, step = (int(x) for x in re.search(r'min="(\d+)" max="(\d+)" step="(\d+)"', page).groups())
        data["choice"] = lo + step * rng.randint(0, (hi - lo) // step)
    else:
        data["choice"] = rng.choice(re.findall(r'name="choice" value="(\w)"', page))
    return c.post(path, data=data)


def play_session(c, rng, n_games=13):  # 10 set games + 3 bonus research games
    for idx in range(n_games):
        assert answer(c, rng, f"/play/{idx}").status_code == 302
    assert c.get(f"/play/{n_games}").status_code == 302  # closes the session


def new_player(app, rng, lang="en"):
    c = app.test_client()
    c.get(f"/lang/{lang}")
    page = c.get("/consent").get_data(as_text=True)
    assert c.post("/consent", data={"csrf": csrf(page), "adult": "yes", "agree": "yes"}).status_code == 302
    play_session(c, rng)
    return c


def next_set(c):
    page = c.get("/profile").get_data(as_text=True)
    assert c.post("/next", data={"csrf": csrf(page)}).status_code == 302


def test_two_visits_profile_share_metrics_export_withdraw(app):
    rng = random.Random(0)
    players = [new_player(app, rng, lang) for lang in ["en", "hi", "es"] * 3]
    for c in players[:6]:
        next_set(c)
        play_session(c, rng)            # week-two set
    next_set(players[0])
    play_session(players[0], rng)       # replay the core set -> "how you changed"

    html = players[0].get("/profile").get_data(as_text=True)
    assert "Steps ahead (11–20 game)" in html and "higher than" in html and "same" in html
    share = re.search(r'value="(http://localhost/s/[\w-]+)"', html).group(1)
    page = app.test_client().get(share)
    assert page.status_code == 200 and b"og:image" in page.data and b"/card/" in page.data
    png = app.test_client().get(share.replace("/s/", "/card/") + ".png")
    assert png.status_code == 200 and png.data[:8] == b"\x89PNG\r\n\x1a\n"

    hi_html = players[1].get("/profile").get_data(as_text=True)
    assert "आपकी रणनीतिक प्रोफ़ाइल" in hi_html

    assert app.test_client().get("/admin/metrics?token=wrong").status_code == 403
    m = app.test_client().get("/admin/metrics?token=test-token").get_data(as_text=True)
    assert "Gate 2 metrics" in m and ">9<" in m

    exp = app.test_client().get("/export.csv?token=test-token").get_data(as_text=True)
    assert exp.count("\n") == 1 + 9 * 13 + 6 * 13 + 13
    page = players[0].get("/withdraw").get_data(as_text=True)
    assert players[0].post("/withdraw", data={"csrf": csrf(page)}).status_code == 200
    exp = app.test_client().get("/export.csv?token=test-token").get_data(as_text=True)
    assert exp.count("\n") == 1 + 8 * 13 + 5 * 13


def test_consent_csrf_and_input_validation(app):
    c = app.test_client()
    assert c.get("/play/0").status_code == 302
    page = c.get("/consent").get_data(as_text=True)
    assert "need to confirm" in c.post("/consent", data={"csrf": csrf(page), "agree": "yes"}).get_data(as_text=True)
    assert c.post("/consent", data={"csrf": "bad", "adult": "yes", "agree": "yes"}).status_code == 400
    c.post("/consent", data={"csrf": csrf(page), "adult": "yes", "agree": "yes"})
    page = c.get("/play/1").get_data(as_text=True)   # ultimatum offer: steps of 10
    assert c.post("/play/1", data={"csrf": csrf(page), "choice": "35", "rt_ms": 2000}).status_code == 400
    assert c.post("/play/1", data={"csrf": csrf(page), "choice": "40", "rt_ms": 2000}).status_code == 302


def test_translations_complete_and_placeholders_match():
    fmt = string.Formatter()
    for lang in LANGS:
        assert missing_keys(lang) == [], lang
        for key, en in STRINGS["en"].items():
            want = {f for _, f, _, _ in fmt.parse(en) if f}
            got = {f for _, f, _, _ in fmt.parse(STRINGS[lang][key]) if f}
            assert want == got, (lang, key)
    for g in GAMES:
        assert f"game.{g.id}.title" in STRINGS["en"]
        if g.kind == "number":
            for role in g.roles:
                assert f"game.{g.id}.{role}" in STRINGS["en"]


def test_game_sets_and_payoff_structure():
    assert len(GAMES) == 20 and len(games_in_set("core")) == 10 and len(games_in_set("week2")) == 10
    # prisoner's dilemma: defect (1) dominates; mutual cooperation beats mutual defection
    assert all(payoff("pd", "row", 1, j) > payoff("pd", "row", 0, j) for j in (0, 1))
    assert payoff("pd", "row", 0, 0) > payoff("pd", "row", 1, 1)
    # stag hunt: two pure equilibria; hawk-dove: no dominant action
    assert payoff("stag", "row", 0, 0) > payoff("stag", "row", 1, 0) and payoff("stag", "row", 1, 1) > payoff("stag", "row", 0, 1)
    # symmetric games really are symmetric
    for g in ("pd", "stag", "hawk_dove"):
        m = MATRICES[g]
        assert all(m["row"][i][j] == m["col"][j][i] for i in range(2) for j in range(2))
    # 11-20 game: asking one less than the other earns the bonus
    assert payoff("money_request", "player", 19, 20) == 39 and payoff("money_request", "player", 20, 20) == 20
    assert payoff("travelers", "claimant", 50, 80) == 60 and payoff("travelers", "claimant", 80, 50) == 40


def test_normal_forms_have_right_shapes():
    for gid in ("ultimatum_offer", "trust_send", "bertrand", "money_request", "travelers", "pd"):
        g = BY_ID[gid]
        mine, theirs, own, other = normal_form(gid, g.roles[0])
        assert len(own) == len(mine) and len(own[0]) == len(theirs)
        assert len(other) == len(theirs) and len(other[0]) == len(mine)


def test_arena_export_becomes_benchmark_records(app, tmp_path):
    from hsb.data import arena as arena_data
    from hsb.models.neural import HybridQLkV1
    rng = random.Random(3)
    players = [new_player(app, rng) for _ in range(24)]
    for c in players:
        next_set(c)
        play_session(c, rng)
    path = tmp_path / "export.csv"
    path.write_bytes(app.test_client().get("/export.csv?token=test-token").data)
    recs = arena_data.load(path, min_choices=5)
    fams = {r.family for r in recs}
    assert {"ultimatum", "trust", "pricing", "undercutting", "matrix"} <= fams
    sizes = {r.game.own.shape for r in recs}
    assert (11, 11) in sizes and (2, 2) in sizes and (10, 10) in sizes
    for r in recs:
        assert abs(r.freq.sum() - 1) < 1e-9 and r.freq.shape[0] == r.game.own.shape[0]
    model = HybridQLkV1(hidden=8, max_steps=50).fit(recs)       # any-size model trains on all of it
    preds = model.predict(recs)
    assert all(abs(p.sum() - 1) < 1e-6 for p in preds)


def test_evaluate_arena_script_runs(app, tmp_path):
    import subprocess, sys
    rng = random.Random(5)
    for _ in range(12):
        c = new_player(app, rng)
        next_set(c)
        play_session(c, rng)
    path = tmp_path / "export.csv"
    path.write_bytes(app.test_client().get("/export.csv?token=test-token").data)
    from hsb.data import arena as arena_data
    from hsb.models.structural import QuantalLevelK
    recs = arena_data.load(path, min_choices=3)
    preds = QuantalLevelK().fit(recs).predict(recs)            # classic models now handle mixed sizes
    assert len(preds) == len(recs) and all(abs(p.sum() - 1) < 1e-9 for p in preds)


def test_bonus_games_calibration_first_then_research(app):
    import arena.app as appmod
    rng = random.Random(9)
    players = [new_player(app, rng) for _ in range(4)]
    with app.app_context():
        rows = appmod.db().execute("SELECT game_id FROM choices WHERE game_id LIKE 'cal-%' OR game_id LIKE 'res-%'").fetchall()
        bonus = [r[0] for r in rows]
    assert len(bonus) == 4 * 3 and all(g.startswith("cal-") for g in bonus)   # calibration fills first
    assert len(set(bonus)) == 12                                               # least-played: all 12 games covered
    old = appmod.CAL_TARGET
    appmod.CAL_TARGET = 0                                                      # calibration "full"
    try:
        c = new_player(app, rng)
        with app.app_context():
            last = appmod.db().execute("SELECT bonus FROM sessions ORDER BY id DESC LIMIT 1").fetchone()[0]
        assert all(g.startswith("res-") for g in json.loads(last))
    finally:
        appmod.CAL_TARGET = old


def test_loader_reads_pool_games(app, tmp_path):
    from hsb.data import arena as arena_data
    rng = random.Random(11)
    for _ in range(8):
        new_player(app, rng)
    path = tmp_path / "export.csv"
    path.write_bytes(app.test_client().get("/export.csv?token=test-token").data)
    cal = arena_data.load(path, min_choices=1, kinds=("calibration",))
    assert cal and all(r.extra["kind"] == "calibration" and r.game.own.shape == (2, 2) for r in cal)
    assert all(r.game.meta["source_game"].startswith("g2p2k-") for r in cal)
