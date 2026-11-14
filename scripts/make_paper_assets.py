"""Generate every number, table and figure in the paper from the result files, so nothing
in the paper is typed by hand.

    python scripts/make_paper_assets.py

Writes paper/generated/*.tex (tables and \\newcommand macros) and paper/figures/*.pdf.
Re-run after any result changes, then rebuild the PDF (see paper/README.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GEN, FIG = ROOT / "paper" / "generated", ROOT / "paper" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

REF = "Hybrid v0 (structural + residual)"
SHORT = {
    "Uniform": "Uniform", "Nash (+ error)": "Nash + error", "Logit QRE": "Logit QRE",
    "Quantal level-k": "Quantal level-$k$", "Cognitive hierarchy": "Cognitive hierarchy",
    "ElementaryNet-style QLk (reimplementation)": "ElementaryNet-style$^\\dagger$",
    "ElementaryNet-style QCH (closer reimplementation)": "ElementaryNet, closer$^\\dagger$",
    "MLP (black box)": "MLP (black box)", "Hybrid v0 (structural only)": "Hybrid, structural only",
    "Hybrid v0 (structural + residual)": "\\textbf{Hybrid (ours)}",
    "Hybrid v1 (any size, structural + residual)": "Hybrid, any-size (ours)",
}
FAM = {"both_dominant": "Both dominant", "self_dominant": "Own dominant",
       "other_dominant": "Opponent dominant", "coordination": "Coordination"}


def nll(df):
    return -(df.f0 * np.log(df.p0.clip(1e-9)) + df.f1 * np.log(df.p1.clip(1e-9)))


def boot_ci(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    S = json.loads((ROOT / "results" / "summary.json").read_text())["results"]
    macros = []

    def macro(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # ---- Table 1: all models x three protocols
    order = [m for m in SHORT if m in S["family"]["overall"]]
    rows = []
    for m in order:
        cells = []
        for proto in ("random", "kfold", "family"):
            s = S[proto]["overall"].get(m)
            cells += ["--", "--"] if s is None else [f"{s['nll']:.4f}", f"{s['completeness']:.3f}"]
        rows.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
    (GEN / "table_main.tex").write_text(
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Random problems} & \\multicolumn{2}{c}{Held-out games} & \\multicolumn{2}{c}{Held-out family} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Model & NLL & Compl. & NLL & Compl. & NLL & Compl. \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")

    for proto, tag in (("kfold", "Games"), ("family", "Family")):
        o = S[proto]["overall"]
        macro(f"nll{tag}Hybrid", f"{o[REF]['nll']:.4f}")
        for key, name in (("Elem", "ElementaryNet-style QLk (reimplementation)"), ("MLP", "MLP (black box)"),
                          ("QLk", "Quantal level-k")):
            d = o[name]["reference_minus_this"]
            macro(f"nll{tag}{key}", f"{o[name]['nll']:.4f}")
            macro(f"diff{tag}{key}", f"{-d['mean']:.4f}")
            macro(f"ci{tag}{key}", f"[{-d['ci95'][1]:.4f}, {-d['ci95'][0]:.4f}]")
        macro(f"compl{tag}Hybrid", f"{o[REF]['completeness']:.3f}")
        macro(f"compl{tag}QLk", f"{o['Quantal level-k']['completeness']:.3f}")
    closer = S["family"]["overall"].get("ElementaryNet-style QCH (closer reimplementation)")
    if closer:
        d = closer["reference_minus_this"]
        macro("nllFamilyElemQCH", f"{closer['nll']:.4f}")
        macro("diffFamilyElemQCH", f"{-d['mean']:.4f}")
        macro("ciFamilyElemQCH", f"[{-d['ci95'][1]:.4f}, {-d['ci95'][0]:.4f}]")
    macro("rTwoRandomMLP", f"{S['random']['overall']['MLP (black box)']['r2']:.3f}")

    # ---- Figure 1: per held-out family, hybrid minus each strong baseline (negative = hybrid better)
    df = pd.read_csv(ROOT / "results" / "predictions" / "family.csv.gz")
    df["nll"] = nll(df)
    piv = df.pivot_table(index=["split", "problem_id"], columns="model", values="nll")
    fams = list(FAM)
    comps = [("ElementaryNet-style QLk (reimplementation)", "ElementaryNet-style"), ("MLP (black box)", "MLP"),
             ("Quantal level-k", "Quantal level-k")]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for k, (name, label) in enumerate(comps):
        means, lo, hi = [], [], []
        for f in fams:
            sub = piv.loc[piv.index.get_level_values(0) == f"holdout-{f}"]
            d = (sub[REF] - sub[name]).to_numpy()
            a, b = boot_ci(d)
            means.append(d.mean()); lo.append(d.mean() - a); hi.append(b - d.mean())
        x = np.arange(len(fams)) + (k - 1) * 0.24
        ax.errorbar(x, means, yerr=[lo, hi], fmt="o", capsize=3, label=f"vs {label}", ms=5)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(np.arange(len(fams)), [FAM[f] for f in fams])
    ax.set_ylabel("NLL difference\n(hybrid − baseline)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "family_differences.pdf")
    plt.close(fig)

    # ---- Figure 2: what the structural core learned
    ex = json.loads((ROOT / "results" / "hybrid_explained.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    left = np.zeros(len(fams))
    shades = ["0.82", "#7aa2ff", "#2b59c3", "#16306b"]
    for k in range(4):
        vals = np.array([ex[f]["level_shares"][f"L{k}"] for f in fams])
        ax.barh([FAM[f] for f in fams], vals, left=left, color=shades[k], label=f"Level {k}")
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average share of players at each reasoning level")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "level_shares.pdf")
    plt.close(fig)
    for f in fams:
        tag = "".join(w.title() for w in f.split("_"))
        l2 = ex[f]["level_shares"]["L2"] + ex[f]["level_shares"]["L3"]
        macro(f"deep{tag}", f"{100 * l2:.0f}")
        macro(f"prec{tag}", f"{ex[f]['precision_own']:.2f}")

    # ---- Table: LLM zero-shot baselines (whatever has been run)
    rows, worse = [], []
    for p in sorted((ROOT / "results").glob("llm_baseline_*.json")):
        r = json.loads(p.read_text())
        name = next(k for k in r["llm"] if k.startswith("LLM"))
        s = r["llm"][name]
        q = r.get("Quantal level-k")
        h = r.get(REF)
        if not q or s.get("n_problems", 0) < 50:
            continue
        d = q["llm_minus_this"]
        model = name.split("(")[1].split(",")[0].split("/")[-1]
        worse.append(d["ci95"][0] > 0)
        rows.append(f"\\texttt{{{model}}} & {s['n_problems']} & {s['nll']:.4f} & {q['nll']:.4f} & "
                    f"{h['nll']:.4f} & {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] \\\\")
    (GEN / "table_llm.tex").write_text(
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccc}\n\\toprule\nLLM (zero-shot) & Problems & LLM NLL & QL$k$ NLL & Hybrid NLL & LLM $-$ QL$k$ [95\\% CI] \\\\\n\\midrule\n"
        + ("\n".join(rows) if rows else "\\multicolumn{6}{c}{(not yet run)} \\\\") + "\n\\bottomrule\n\\end{tabular}}\n")

    if rows and all(worse):
        lead = "Both models are" if len(rows) == 2 else ("The model is" if len(rows) == 1 else f"All {({3: 'three', 4: 'four', 5: 'five'}).get(len(rows), len(rows))} models are")
        sentence = (f"{lead} significantly worse than quantal level-$k$, a theory with four free "
                    "parameters, and further still from the hybrid: asked to role-play people, they are poor stand-ins "
                    "for them, in line with \\citet{wang2026discovering}.")
    elif rows:
        sentence = ("The models are not uniformly worse than quantal level-$k$; see the confidence intervals "
                    "in Table~\\ref{tab:llm}.")
    else:
        sentence = "These runs are not yet complete."
    (GEN / "llm_sentence.tex").write_text(sentence + "\n")
    # ---- Table + macros: confirmatory transfer test (H1, protocol v3)
    H = json.loads((ROOT / "results" / "h1_confirmatory.json").read_text())
    hs, test = H["summary"], "Hybrid v1 (any size, structural + residual)"
    sec = H["secondary"]
    h1_rows = []
    for m in [k for k in SHORT if k in hs]:
        d = (H["primary"].get(m) or sec["comparisons"].get(m))
        diff = f"{d['mean_diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else ""
        h1_rows.append(f"{SHORT[m].replace('Hybrid, any-size (ours)', '\\textbf{Hybrid, any-size (ours)}')} & "
                       f"{hs[m]['nll']:.4f} & {sec['nll_weighted_by_choices'][m]:.4f} & "
                       f"{sec['without_large_games']['nll'][m]:.4f} & {diff} \\\\")
    (GEN / "table_h1.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\nModel & NLL & NLL (by choices) & NLL (no large games) & Hybrid $-$ model [95\\% CI] \\\\\n\\midrule\n"
        + "\n".join(h1_rows) + "\n\\bottomrule\n\\end{tabular}\n")
    macro("hOneProblems", f"{H['n_problems']}")
    macro("hOneVerdict", H["verdict"].lower())
    for tag, m in [("Hybrid", test), ("QLk", "Quantal level-k"), ("CH", "Cognitive hierarchy"),
                   ("Elem", "ElementaryNet-style QLk (reimplementation)"), ("Uniform", "Uniform"),
                   ("ElemQCH", "ElementaryNet-style QCH (closer reimplementation)")]:
        macro(f"hOneNll{tag}", f"{hs[m]['nll']:.3f}")
    for tag, d in [("Elem", H["primary"]["ElementaryNet-style QLk (reimplementation)"]),
                   ("QLk", H["primary"]["Quantal level-k"])]:
        macro(f"hOneDiff{tag}", f"{d['mean_diff']:+.3f}")
        macro(f"hOneCi{tag}", f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
        macro(f"hOneP{tag}", f"{d['p_one_sided']:.3f}")

    (GEN / "macros.tex").write_text("\n".join(macros) + "\n")
    print(f"wrote {len(macros)} macros, 3 tables, 2 figures; LLM rows: {len(rows)}")


if __name__ == "__main__":
    main()
