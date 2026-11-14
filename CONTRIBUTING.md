# Contributing

Thanks for helping. Three kinds of contribution are especially welcome.

## 1. Submit a model to the leaderboard

1. Predict every problem under both protocols, using only the training data each split
   allows (`benchmark/data/splits.json`).
2. Score it: `python benchmark/score.py preds.csv`
3. Save it: `python benchmark/score.py preds.csv --save --name "Team: Model" --url <code or paper> --desc "one line"`
4. Open a pull request with the new `benchmark/submissions/*.json` file and a link to your code.
   LLM entries must include the exact prompt and model version.

We re-run every submission before it appears on the leaderboard.

## 2. Add a dataset

Write a loader in `hsb/data/` that returns `Record`s (see `hsb/data/schema.py`), with the
licence and citation in its docstring, and a test. Datasets from outside Western, educated,
rich populations are especially valuable.

## 3. Run a study on the arena

Researchers can run their own games on the arena for free, in exchange for sharing the
anonymised data. Open an issue describing the game and your ethics approval.

## Ground rules

- Every number in a paper or on the leaderboard must come from a script in this repository.
- Never commit participant data. The arena export stays off GitHub.
- Run `python -m pytest -q` before opening a pull request.
