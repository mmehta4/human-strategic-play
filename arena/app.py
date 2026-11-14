"""Arena v1: 20 strategic games in two weekly sets, three languages, a measured profile,
a shareable card, and a metrics page that tracks the Gate 2 targets.

Run locally:
    ARENA_SECRET=change-me ARENA_ADMIN_TOKEN=change-me-too python -m arena.app
then open http://127.0.0.1:5055. In production run it with gunicorn (see DEPLOY.md).

Data rules: 18+ only, explicit consent first, pseudonymous IDs (no names, emails or
IP addresses stored), coarse optional demographics, reaction times and language
logged, players can withdraw and delete their data at any time. No money is involved.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import secrets
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, Response, abort, g, redirect, render_template, request, session, url_for

from . import card, pool
from .games import BY_ID, GAMBLE_GAINS, games_in_set, matrix_lines, option_labels, set_for_session
from .i18n import LANGS, STRINGS, best_language, translate
from .profile import TRAITS, changes, percentile, results, share_same, traits

CONSENT_VERSION = "2026-10-v1"
MIN_RT_MS = 800                      # faster than this is flagged as likely automated
RETURN_GAP_S = 24 * 3600             # a "return" is a new visit at least a day after the first
DB_PATH = Path(os.environ.get("ARENA_DB", Path.home() / "arena-dev.db"))  # never inside iCloud
GATE2 = {"players": 25_000, "decisions": 500_000, "return_rate": 0.30}
POOL = pool.load() if pool.POOL_PATH.exists() else {}
CAL_TARGET = 100                     # choices per calibration game before research games take over

app = Flask(__name__, root_path=str(Path(__file__).resolve().parent))
app.secret_key = os.environ.get("ARENA_SECRET") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=os.environ.get("ARENA_HTTPS") == "1",
                  PERMANENT_SESSION_LIFETIME=timedelta(days=180))

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS participants (
    id TEXT PRIMARY KEY, created_at REAL, consent_version TEXT, age_band TEXT, country TEXT,
    education TEXT, lang TEXT, share_id TEXT UNIQUE, flagged INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, participant_id TEXT, session_no INTEGER, set_id TEXT,
    lang TEXT, started_at REAL, finished_at REAL, bonus TEXT, UNIQUE(participant_id, session_no)
);
CREATE TABLE IF NOT EXISTS choices (
    id INTEGER PRIMARY KEY AUTOINCREMENT, participant_id TEXT, session_no INTEGER, game_id TEXT,
    role TEXT, choice REAL, context TEXT, rt_ms INTEGER, lang TEXT, created_at REAL,
    flagged INTEGER DEFAULT 0, UNIQUE(participant_id, game_id, session_no)
);
CREATE INDEX IF NOT EXISTS choices_game ON choices (game_id, flagged);
"""


# ------------------------------------------------------------------ plumbing

def db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
        g.db.executescript(SCHEMA)
        cols = {r[1] for r in g.db.execute("PRAGMA table_info(sessions)")}
        if "bonus" not in cols:                       # databases created before the research pool
            g.db.execute("ALTER TABLE sessions ADD COLUMN bonus TEXT")
    return g.db


@app.teardown_appcontext
def close_db(_):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def lang() -> str:
    if session.get("lang") not in LANGS:
        session["lang"] = best_language(request.headers.get("Accept-Language"))
    return session["lang"]


def t(key: str, **kw) -> str:
    return translate(lang(), key, **kw)


def csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


app.jinja_env.globals.update(t=t, csrf_token=csrf_token, langs=LANGS, strings=STRINGS, current_lang=lang)


@app.before_request
def before():
    session.permanent = True
    if request.method == "POST" and request.form.get("csrf") != session.get("csrf"):
        abort(400)


@app.after_request
def headers(resp):
    resp.headers.setdefault("Content-Security-Policy",
                            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                            "script-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    return resp


def current_pid() -> str | None:
    pid = session.get("pid")
    if pid and db().execute("SELECT 1 FROM participants WHERE id=?", (pid,)).fetchone():
        return pid
    return None


def latest_session(pid: str):
    return db().execute("SELECT * FROM sessions WHERE participant_id=? ORDER BY session_no DESC LIMIT 1",
                        (pid,)).fetchone()


def pick_bonus(pid: str) -> list[str]:
    """Least-played pool games first: calibration games until each has CAL_TARGET choices,
    then research games. Never repeats a pool game for the same player."""
    if not POOL:
        return []
    counts = Counter({gid: 0 for gid in POOL})
    for gid, n in db().execute("SELECT game_id, COUNT(*) FROM choices WHERE flagged=0 AND "
                               "(game_id LIKE 'cal-%' OR game_id LIKE 'res-%') GROUP BY game_id"):
        counts[gid] = n
    seen = {r[0] for r in db().execute("SELECT game_id FROM choices WHERE participant_id=?", (pid,))}
    for s_row in db().execute("SELECT bonus FROM sessions WHERE participant_id=?", (pid,)):
        seen |= set(json.loads(s_row[0] or "[]"))
    cal = [g for g in POOL if POOL[g]["kind"] == "calibration" and counts[g] < CAL_TARGET and g not in seen]
    res = [g for g in POOL if POOL[g]["kind"] == "research" and g not in seen]
    rng = random.Random(f"{pid}:{len(seen)}")
    picked = []
    for cands in (cal, res):
        rng.shuffle(cands)
        cands.sort(key=lambda g: counts[g])           # stable sort: random among equally-played
        for g in cands:
            if len(picked) < pool.BONUS_PER_VISIT and g not in picked:
                picked.append(g)
    return picked


def start_session(pid: str, session_no: int) -> None:
    db().execute("INSERT INTO sessions (participant_id, session_no, set_id, lang, started_at, bonus) "
                 "VALUES (?,?,?,?,?,?)", (pid, session_no, set_for_session(session_no), lang(), time.time(),
                                          json.dumps(pick_bonus(pid))))
    db().commit()


def role_for(pid: str, game_id: str, roles: list[str]) -> str:
    """Stable per player and game, so a replay is the same decision problem."""
    return random.Random(f"{pid}:{game_id}").choice(roles)


def value_for(pid: str, session_no: int, game_id: str) -> int:
    return random.Random(f"{pid}:{session_no}:{game_id}").randint(1, 100)


# ------------------------------------------------------------------ pages

@app.get("/")
def index():
    n = db().execute("SELECT COUNT(DISTINCT participant_id) FROM choices WHERE flagged=0").fetchone()[0]
    pid = current_pid()
    last = latest_session(pid) if pid else None
    return render_template("index.html", n_players=n, n_games=len(games_in_set("core")),
                           returning=bool(last and last["finished_at"]), in_progress=bool(last and not last["finished_at"]))


@app.get("/lang/<code>")
def set_lang(code: str):
    if code in LANGS:
        session["lang"] = code
    ref = request.referrer
    if ref and urlparse(ref).netloc == request.host:
        return redirect(ref)
    return redirect(url_for("index"))


@app.route("/consent", methods=["GET", "POST"])
def consent():
    if current_pid():
        return redirect(url_for("play", idx=0))
    if request.method == "POST":
        f = request.form
        if f.get("adult") != "yes" or f.get("agree") != "yes":
            return render_template("consent.html", error=t("consent.error"))
        pid = uuid.uuid4().hex
        db().execute("INSERT INTO participants (id, created_at, consent_version, age_band, country, education, "
                     "lang, share_id) VALUES (?,?,?,?,?,?,?,?)",
                     (pid, time.time(), CONSENT_VERSION, f.get("age_band") or None,
                      (f.get("country") or "").strip()[:56] or None, f.get("education") or None,
                      lang(), secrets.token_urlsafe(9)))
        db().commit()
        session["pid"] = pid
        start_session(pid, 1)
        return redirect(url_for("play", idx=0))
    return render_template("consent.html", error=None)


@app.post("/next")
def next_set():
    pid = current_pid()
    if pid is None:
        return redirect(url_for("consent"))
    last = latest_session(pid)
    if last and not last["finished_at"]:
        return redirect(url_for("play", idx=0))
    start_session(pid, (last["session_no"] if last else 0) + 1)
    return redirect(url_for("play", idx=0))


@app.route("/play/<int:idx>", methods=["GET", "POST"])
def play(idx: int):
    pid = current_pid()
    if pid is None:
        return redirect(url_for("consent"))
    sess = latest_session(pid)
    if sess is None or sess["finished_at"]:
        return redirect(url_for("profile"))
    games = games_in_set(sess["set_id"])
    bonus = [g for g in json.loads(sess["bonus"] or "[]") if g in POOL]
    total = len(games) + len(bonus)
    if idx >= total:
        db().execute("UPDATE sessions SET finished_at=? WHERE id=?", (time.time(), sess["id"]))
        db().commit()
        return redirect(url_for("profile"))

    is_bonus = idx >= len(games)
    if is_bonus:
        pg = POOL[bonus[idx - len(games)]]
        game_id, kind, role = pg["id"], "choice", role_for(pid, pg["id"], ["row", "col"])
        labels = pool.options(pg, role)
        game = None
    else:
        game = games[idx]
        game_id, kind = game.id, game.kind
        role = role_for(pid, game.id, game.roles)
        labels = option_labels(role) if kind == "choice" else []
    value = value_for(pid, sess["session_no"], game_id)

    if request.method == "POST":
        f = request.form
        rt = int(f.get("rt_ms") or 0)
        flagged = int(bool(f.get("website")) or rt < MIN_RT_MS)  # honeypot field or too fast
        context = {}
        if kind == "gambles":
            accepted = [f.get(f"g{i}") == "play" for i in range(len(GAMBLE_GAINS))]
            choice, context = float(sum(accepted)), {"accepted": accepted}
        elif kind == "choice":
            if f.get("choice") not in labels:
                abort(400)
            choice = float(labels.index(f["choice"]))
            if is_bonus:
                context = {"pool_sha": pool.SEED}
        else:
            try:
                choice = float(f["choice"])
            except (KeyError, ValueError):
                abort(400)
            if not game.min <= choice <= game.max or round((choice - game.min) / game.step, 6) % 1:
                abort(400)
            if game.id in ("auction", "second_price"):
                context = {"value": value}
        db().execute("INSERT OR REPLACE INTO choices (participant_id, session_no, game_id, role, choice, context, "
                     "rt_ms, lang, created_at, flagged) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (pid, sess["session_no"], game_id, role, choice, json.dumps(context), rt, lang(),
                      time.time(), flagged))
        db().commit()
        return redirect(url_for("play", idx=idx + 1))

    if is_bonus:
        text, title = pool.lines(pg, role, t), t("game.bonus.title")
    elif kind == "choice":
        text, title = matrix_lines(game.id, role, t), t(f"game.{game.id}.title")
    else:
        text = [t(f"game.{game.id}.{role}", value=value) if game.id in ("auction", "second_price")
                else t(f"game.{game.id}.{role}")]
        title = t(f"game.{game.id}.title")
    return render_template("play.html", game=game, kind=kind, title=title, is_bonus=is_bonus, role=role,
                           idx=idx, total=total, text=text, gains=GAMBLE_GAINS, options=labels)


def _rows_to_choices(rows):
    out = {}
    for r in rows:
        out[r["game_id"]] = {"choice": r["choice"], "role": r["role"], "context": json.loads(r["context"])}
    return out


def build_profile(pid: str) -> dict:
    """Latest choice per game for the profile; first choice per game for "how you changed"."""
    mine_rows = db().execute("SELECT * FROM choices WHERE participant_id=? ORDER BY session_no", (pid,)).fetchall()
    latest = _rows_to_choices(mine_rows)                     # later sessions overwrite earlier ones
    first = _rows_to_choices(reversed(mine_rows))
    plays = Counter(r["game_id"] for r in mine_rows)
    replayed = {gid for gid, n in plays.items() if n > 1}
    others = db().execute(
        "SELECT c.* FROM choices c JOIN (SELECT participant_id, game_id, MAX(session_no) s FROM choices "
        "WHERE flagged=0 GROUP BY participant_id, game_id) m ON c.participant_id=m.participant_id "
        "AND c.game_id=m.game_id AND c.session_no=m.s WHERE c.flagged=0").fetchall()
    pool = defaultdict(lambda: defaultdict(list))
    by_player = defaultdict(dict)
    for r in others:
        ctx = json.loads(r["context"])
        pool[r["game_id"]][r["role"]].append((r["participant_id"], r["choice"], ctx))
        by_player[r["participant_id"]][r["game_id"]] = {"choice": r["choice"], "role": r["role"], "context": ctx}
    my_traits = traits(latest)
    everyone = [traits(c) for p, c in by_player.items() if p != pid]
    items = []
    for key, v in my_traits.items():
        kind, top = TRAITS[key]
        pop = [e.get(key) for e in everyone]
        if kind == "yes":
            same = share_same(v, pop)
            compare = t("profile.same_as", p=same) if same is not None else ""
        else:
            pct = percentile(v, pop)
            compare = t("profile.higher_than", p=pct) if pct is not None else ""
        items.append({"key": key, "name": t(f"trait.{key}.name"), "why": t(f"trait.{key}.why"),
                      "value": _display(key, v, kind, top), "bar": _bar(v, kind, top), "compare": compare})
    delta = changes({k: first[k] for k in replayed}, {k: latest[k] for k in replayed})
    return {"items": items, "results": results(pid, latest, pool), "played": list(latest),
            "changes": [{"name": t(f"trait.{k}.name"), "delta": d} for k, d in delta.items()],
            "n_players": len(by_player), "traits": my_traits}


def _display(key, v, kind, top):
    if v is None:
        return t("profile.above3")
    if kind == "steps":
        return t("profile.step" if v == 1 else "profile.steps", v=f"{v:g}")
    if kind == "x":
        return t("profile.x_val", v=f"{v:g}")
    if kind == "pct":
        return f"{round(v)}%"
    if kind == "of":
        return t("profile.of", v=f"{v:g}", top=top)
    return t("profile.yes") if v else t("profile.no")


def _bar(v, kind, top):
    if kind == "yes":
        return None
    frac = 1.0 if v is None else v / top
    return max(0, min(100, round(100 * frac)))


@app.get("/profile")
def profile():
    pid = current_pid()
    if pid is None:
        return redirect(url_for("consent"))
    sess = latest_session(pid)
    if sess and not sess["finished_at"]:
        return redirect(url_for("play", idx=0))
    p = build_profile(pid)
    share_id = db().execute("SELECT share_id FROM participants WHERE id=?", (pid,)).fetchone()[0]
    return render_template("profile.html", p=p, games=[BY_ID[gid] for gid in p["played"] if gid in BY_ID], pid=pid,
                           share_url=url_for("share", share_id=share_id, _external=True))


def _participant_by_share(share_id: str):
    row = db().execute("SELECT id FROM participants WHERE share_id=?", (share_id,)).fetchone()
    if row is None:
        abort(404)
    return row["id"]


@app.get("/s/<share_id>")
def share(share_id: str):
    pid = _participant_by_share(share_id)
    p = build_profile(pid)
    return render_template("share.html", p=p, share_id=share_id,
                           card_url=url_for("card_png", share_id=share_id, _external=True),
                           page_url=url_for("share", share_id=share_id, _external=True))


@app.get("/card/<share_id>.png")
def card_png(share_id: str):
    pid = _participant_by_share(share_id)
    png = card.render(build_profile(pid)["traits"], TRAITS, translate("en", "site.name"))
    return Response(png, mimetype="image/png", headers={"Cache-Control": "public, max-age=600"})


@app.route("/withdraw", methods=["GET", "POST"])
def withdraw():
    pid = current_pid()
    if request.method == "POST" and pid:
        for table, col in (("choices", "participant_id"), ("sessions", "participant_id"), ("participants", "id")):
            db().execute(f"DELETE FROM {table} WHERE {col}=?", (pid,))
        db().commit()
        keep_lang = session.get("lang")
        session.clear()
        session["lang"] = keep_lang
        return render_template("withdrawn.html")
    return render_template("withdraw.html", has_data=bool(pid))


def _admin():
    token = os.environ.get("ARENA_ADMIN_TOKEN")
    if not token or not secrets.compare_digest(request.args.get("token", ""), token):
        abort(403)


@app.get("/export.csv")
def export():
    _admin()
    rows = db().execute(
        "SELECT c.participant_id, c.session_no, c.game_id, c.role, c.choice, c.context, c.rt_ms, c.lang, "
        "c.created_at, c.flagged, p.age_band, p.country, p.education, p.consent_version FROM choices c "
        "JOIN participants p ON p.id = c.participant_id ORDER BY c.created_at").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(rows[0].keys() if rows else ["participant_id"])
    w.writerows([tuple(r) for r in rows])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=arena_export.csv"})


def metrics() -> dict:
    now = time.time()
    players = db().execute("SELECT COUNT(DISTINCT participant_id) FROM choices WHERE flagged=0").fetchone()[0]
    decisions = db().execute("SELECT COUNT(*) FROM choices WHERE flagged=0").fetchone()[0]
    flagged = db().execute("SELECT COUNT(*) FROM choices WHERE flagged=1").fetchone()[0]
    sess = db().execute("SELECT participant_id, started_at FROM sessions ORDER BY started_at").fetchall()
    first, returned = {}, set()
    for r in sess:
        p, t0 = r["participant_id"], r["started_at"]
        if p not in first:
            first[p] = t0
        elif RETURN_GAP_S <= t0 - first[p] <= 30 * 86400:
            returned.add(p)
    mature = [p for p, t0 in first.items() if now - t0 >= 30 * 86400]
    rate = (sum(p in returned for p in mature) / len(mature)) if mature else None
    langs = Counter(r[0] for r in db().execute("SELECT lang FROM participants"))
    countries = Counter(r[0] or "(not given)" for r in db().execute("SELECT country FROM participants")).most_common(10)
    return {"players": players, "decisions": decisions,
            "flagged_share": flagged / (decisions + flagged) if decisions + flagged else 0.0,
            "return_rate_30d": rate, "mature_cohort": len(mature),
            "returned_so_far": len(returned), "languages": dict(langs), "countries": countries,
            "gate": GATE2}


@app.get("/admin/metrics")
def admin_metrics():
    _admin()
    return render_template("metrics.html", m=metrics())


@app.get("/privacy")
def privacy():
    return render_template("privacy.html", version=CONSENT_VERSION)


@app.get("/healthz")
def healthz():
    db().execute("SELECT 1")
    return {"ok": True}


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5055)))
