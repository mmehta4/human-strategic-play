# Paper draft

Every number, table and figure comes from the result files. Nothing in `main.tex` is typed by hand.

Rebuild after any result changes:
```bash
cd stage1
~/.venvs/hsb-stage1/bin/python scripts/make_paper_assets.py   # tables, figures, numbers
cd paper && tectonic -X compile main.tex                      # -> main.pdf
```
No LaTeX on your computer? Upload `main.tex`, `refs.bib`, `generated/` and `figures/` to Overleaf
(free) and compile there. For arXiv, upload the same files.

Before posting: fill in the author name and email (top of `main.tex`), add your advisor as co-author
if they agree, and have your advisor read the whole draft.
