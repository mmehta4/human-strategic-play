"""Zero-shot LLM baseline on free API tiers (Groq, Google Gemini, NVIDIA build).

All three speak the OpenAI-compatible chat-completions format, so one small client covers
them. Keys come only from environment variables, never from files in this repository:

    GROQ_API_KEY     console.groq.com          e.g. openai/gpt-oss-120b
    GEMINI_API_KEY   aistudio.google.com       e.g. gemini-3.5-flash-lite
    NVIDIA_API_KEY   build.nvidia.com          e.g. nvidia/nemotron-3-ultra-550b-a55b, moonshotai/kimi-k3

Free tiers are rate-limited, so requests are spaced out, 429s are retried with backoff,
and every answer is cached (results/llm_cache.jsonl). A run that hits a daily limit
can simply be re-run the next day and continues where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data.schema import Record
from .llm import PROMPT_VERSION, QUESTION, SYSTEM
from .models.base import Model
from .text import ROW_LABELS, describe_matrix_game

PROVIDERS = {
    "groq": {"base": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY", "min_gap_s": 25},  # free tier: 8,000 tokens/min
    "gemini": {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "env": "GEMINI_API_KEY", "min_gap_s": 4.5},
    "nvidia": {"base": "https://integrate.api.nvidia.com/v1", "env": "NVIDIA_API_KEY", "min_gap_s": 1.6},
}
FORMAT = ('\n\nReply with only a JSON object, no other text, in exactly this form:\n'
          '{"reasoning": "<one or two sentences>", "percent": {LABELS}}')


class DailyLimit(RuntimeError):
    pass


class FreeLLMZeroShot(Model):
    def __init__(self, provider: str, model: str, workers: int = 2, cache_path: Path | None = None,
                 max_retries: int = 8):
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {list(PROVIDERS)}")
        cfg = PROVIDERS[provider]
        self.key = os.environ.get(cfg["env"])
        if not self.key:
            raise RuntimeError(f"Set {cfg['env']} in your environment (never put keys in files here).")
        self.provider, self.model, self.base, self.gap = provider, model, cfg["base"], cfg["min_gap_s"]
        self.workers, self.max_retries = workers, max_retries
        self.name = f"LLM zero-shot ({model}, {provider} free tier)"
        self.cache_path = cache_path or Path(__file__).resolve().parents[1] / "results" / "llm_cache.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row
        self._lock, self._last = threading.Lock(), 0.0
        self.failures, self.refusals, self.stopped = 0, 0, None

    def fit(self, records):
        return self

    def _wait_turn(self):
        with self._lock:
            now = time.time()
            wait = self._last + self.gap - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                              "User-Agent": "hsb-benchmark/0.2 (+research)", "Accept": "application/json"})
        for attempt in range(self.max_retries):
            self._wait_turn()
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code == 429:
                    low = detail.lower()
                    if any(k in low for k in ("per day", "daily", "rpd", "tpd", "quota")):
                        raise DailyLimit(detail) from e
                    if attempt == 0:
                        print(f"[{self.provider}] rate limited: {detail[:200]}", file=sys.stderr)
                    retry = float(e.headers.get("retry-after") or 0) or min(120, 10 * 2 ** attempt)
                    time.sleep(min(retry, 300))
                    continue
                if e.code >= 500:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise RuntimeError(f"{self.provider} HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError):
                time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"{self.provider}: gave up after {self.max_retries} attempts")

    def _ask(self, record: Record):
        m = record.game.n_actions
        labels = ", ".join(f'"{ROW_LABELS[i]}": <number>' for i in range(m))
        prompt = describe_matrix_game(record.game) + "\n\n" + QUESTION + FORMAT.replace("LABELS", labels)
        key = hashlib.sha256(f"{self.provider}|{self.model}|{PROMPT_VERSION}|{SYSTEM}|{prompt}".encode()).hexdigest()
        if key not in self.cache:
            if self.stopped:
                return None
            try:
                    data = self._post({"model": self.model, "temperature": 0,
                                   "messages": [{"role": "system", "content": SYSTEM},
                                                {"role": "user", "content": prompt}]})
            except DailyLimit as e:
                self.stopped = f"daily limit reached: {e}"
                return None
            except RuntimeError as e:          # one stubborn question: skip it, keep the run going
                print(f"[{self.provider}] skipped {record.problem_id}: {str(e)[:200]}", file=sys.stderr)
                self.failures += 1
                return None
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            pct = _parse(text, m)
            if pct is None:
                self.failures += 1
                return None
            row = {"key": key, "problem_id": record.problem_id, "provider": self.provider,
                   "model": self.model, "answer": {"percent": pct}, "raw": text[-600:]}
            with self._lock:
                self.cache[key] = row
                with self.cache_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
        pct = self.cache[key]["answer"]["percent"]
        p = np.array([max(float(pct.get(ROW_LABELS[i], 0)), 0.0) for i in range(m)])
        p = p / p.sum() if p.sum() > 0 else np.full(m, 1 / m)
        p = np.clip(p, 1e-3, 1)       # the same generous floor as the Claude baseline
        return p / p.sum()

    def predict(self, records):
        """Returns predictions; problems the model couldn't answer (parse failures) are None."""
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._ask, records))


def _parse(text: str, m: int):
    """Pull {"percent": {...}} out of a reply, tolerating markdown fences or extra prose."""
    for cand in [text] + re.findall(r"\{.*\}", text, flags=re.S):
        try:
            obj = json.loads(cand.strip().strip("`").removeprefix("json").strip())
        except (json.JSONDecodeError, AttributeError):
            continue
        pct = obj.get("percent") if isinstance(obj, dict) else None
        if isinstance(pct, dict) and all(ROW_LABELS[i] in pct for i in range(m)):
            try:
                return {k: float(v) for k, v in pct.items()}
            except (TypeError, ValueError):
                continue
    return None
