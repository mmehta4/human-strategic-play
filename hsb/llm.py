"""Zero-shot LLM baseline: ask a frontier model what share of people choose each
option. This is the "LLM as a stand-in for people" approach the company argues
against, so we measure it on exactly the same problems and metric as everything else.

Costs real money and needs Anthropic credentials (ANTHROPIC_API_KEY, or a profile
from `ant auth login`). Every answer is cached in results/llm_cache.jsonl, keyed by
model + prompt, so re-running never pays twice for the same question.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data.schema import Record
from .models.base import Model
from .text import ROW_LABELS, describe_matrix_game

PROMPT_VERSION = "v1"
SYSTEM = (
    "You are an expert in experimental economics. You predict how real people, not "
    "ideal rational agents, behave in one-shot strategic games run online with "
    "ordinary adult participants recruited from a research panel, where choices "
    "carry small real-money bonuses."
)
QUESTION = (
    "Predict the percentage of participants in the role described (\"you\") who would choose "
    "each of their options. Give your best calibrated estimate of the population shares, not "
    "the single most likely choice. Percentages must sum to 100."
)


def _schema(m: int) -> dict:
    props = {ROW_LABELS[i]: {"type": "number"} for i in range(m)}
    return {
        "type": "object",
        "properties": {"reasoning": {"type": "string"}, "percent": {
            "type": "object", "properties": props, "required": list(props), "additionalProperties": False}},
        "required": ["reasoning", "percent"],
        "additionalProperties": False,
    }


class LLMZeroShot(Model):
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium",
                 cache_path: Path | None = None, workers: int = 4):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.workers = model, effort, workers
        self.name = f"LLM zero-shot ({model})"
        self.cache_path = cache_path or Path(__file__).resolve().parents[1] / "results" / "llm_cache.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row
        self.refusals = 0

    def fit(self, records):
        return self  # zero-shot: nothing is learned from data

    def _ask(self, record: Record) -> np.ndarray:
        import anthropic
        prompt = describe_matrix_game(record.game) + "\n\n" + QUESTION
        key = hashlib.sha256(f"{self.model}|{self.effort}|{PROMPT_VERSION}|{SYSTEM}|{prompt}".encode()).hexdigest()
        m = record.game.n_actions
        if key not in self.cache:
            try:
                resp = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"effort": self.effort,
                                   "format": {"type": "json_schema", "schema": _schema(m)}},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except anthropic.RateLimitError:
                raise
            except anthropic.APIStatusError as e:
                raise RuntimeError(f"API error {e.status_code} on {record.problem_id}: {e.message}") from e
            if resp.stop_reason == "refusal":
                self.refusals += 1
                return np.full(m, 1 / m)  # counted, and scored as a uniform guess
            text = next(b.text for b in resp.content if b.type == "text")
            row = {"key": key, "problem_id": record.problem_id, "model": resp.model,
                   "answer": json.loads(text)}
            self.cache[key] = row
            with self.cache_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        pct = self.cache[key]["answer"]["percent"]
        p = np.array([max(float(pct[ROW_LABELS[i]]), 0.0) for i in range(m)])
        p = p / p.sum() if p.sum() > 0 else np.full(m, 1 / m)
        return np.clip(p, 1e-3, 1)  # an LLM saying 0% must not get infinite loss
                                    # forever; this floor is generous to the LLM

    def predict(self, records):
        with ThreadPoolExecutor(self.workers) as pool:
            preds = list(pool.map(self._ask, records))
        p = np.array(preds)
        return p / p.sum(1, keepdims=True)
