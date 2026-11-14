"""Build the public benchmark + leaderboard website (static HTML, no server needed).

    python scripts/build_leaderboard.py

Reads site/config.json (name, links), results/summary.json (our baselines) and
benchmark/submissions/*.json (outside teams, written by `benchmark/score.py --save`),
and writes site/index.html. Host the site/ folder on GitHub Pages, Netlify or Cloudflare Pages.
"""
from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def rows():
    out = []
    summary = json.loads((ROOT / "results" / "summary.json").read_text())["results"]
    for model, s in summary["family"]["overall"].items():
        k = summary["kfold"]["overall"].get(model)
        out.append({"name": model, "who": "Our model" if model.startswith("Hybrid") else "Reference baseline", "url": "",
                    "family": s["nll"], "family_c": s["completeness"], "kfold": k["nll"] if k else None,
                    "ours": model.startswith("Hybrid"), "external": False})
    for f in sorted((ROOT / "benchmark" / "submissions").glob("*.json")):
        sub = json.loads(f.read_text())
        sc = sub["scores"]
        if "family" not in sc:
            continue
        out.append({"name": sub["name"], "who": sub.get("description", ""), "url": sub.get("url", ""),
                    "family": sc["family"]["nll"], "family_c": sc["family"]["completeness"],
                    "kfold": sc.get("kfold", {}).get("nll"), "ours": False, "external": True})
    return sorted(out, key=lambda r: r["family"])


def table(rs):
    e = html.escape
    body = []
    for i, r in enumerate(rs, 1):
        name = f'<a href="{e(r["url"])}">{e(r["name"])}</a>' if r["url"] else e(r["name"])
        tag = ' <span class="tag">submitted</span>' if r["external"] else ""
        kf = f'{r["kfold"]:.4f}' if r["kfold"] is not None else "–"
        body.append(f'<tr{" class=ours" if r["ours"] else ""}><td class="num">{i}</td><td>{name}{tag}'
                    f'<div class="sub">{e(r["who"])}</div></td><td class="num">{r["family"]:.4f}</td>'
                    f'<td class="num">{r["family_c"]:.3f}</td><td class="num">{kf}</td></tr>')
    return "\n".join(body)


def main():
    cfg = json.loads((SITE / "config.json").read_text())
    rs = rows()
    page = (SITE / "template.html").read_text()
    for k, v in {**cfg, "rows": table(rs), "updated": date.today().isoformat(),
                 "n_models": str(len(rs))}.items():
        page = page.replace("{{" + k + "}}", v if k == "rows" else html.escape(v))
    (SITE / "index.html").write_text(page)
    print(f"wrote {SITE / 'index.html'} with {len(rs)} models")


if __name__ == "__main__":
    main()
