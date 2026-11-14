"""Translations. English is the source of truth; any key missing in another language
falls back to English, so a half-finished translation never breaks a page."""
from __future__ import annotations

import json
from pathlib import Path

DIR = Path(__file__).resolve().parent
LANGS = ["en", "hi", "es"]
STRINGS = {code: json.loads((DIR / f"{code}.json").read_text(encoding="utf-8")) for code in LANGS}


def translate(lang: str, key: str, **kw) -> str:
    text = STRINGS.get(lang, {}).get(key) or STRINGS["en"].get(key, key)
    return text.format(**kw) if kw else text


def best_language(accept_language: str | None) -> str:
    for part in (accept_language or "").split(","):
        code = part.split(";")[0].strip().lower()[:2]
        if code in LANGS:
            return code
    return "en"


def missing_keys(lang: str) -> list[str]:
    return sorted(set(STRINGS["en"]) - set(STRINGS[lang]))
