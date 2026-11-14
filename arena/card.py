"""Shareable profile card: a 1200x630 PNG (the size X, LinkedIn and WhatsApp preview).
It shows up to four measured traits and nothing that identifies the player.

The card is rendered in English for now: Pillow's bundled font has no Devanagari, so
Hindi text needs a bundled font (e.g. Noto Sans Devanagari) before cards can be translated.
"""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from .i18n import translate

W, H = 1200, 630
BG, INK, MUTED, LINE, ACCENT = (250, 249, 246), (29, 29, 27), (107, 106, 102), (228, 226, 220), (43, 89, 195)
ORDER = ["reasoning_depth", "depth_1120", "min_accept_pct", "trust_pct", "cooperation_pct",
         "fair_offer_pct", "reciprocity_pct", "loss_aversion", "dictator_pct", "effort"]


def _font(size):
    return ImageFont.load_default(size=size)


def display_value(key: str, v, kind: str, top) -> str:
    if v is None:
        return "above 3" if key == "loss_aversion" else "-"
    if kind == "steps":
        return f"{v:g} steps"
    if kind == "x":
        return f"{v:g}x"
    if kind == "pct":
        return f"{round(v)}%"
    if kind == "of":
        return f"{v:g} of {top}"
    return "Yes" if v else "No"


def render(traits: dict, kinds: dict, site_name: str) -> bytes:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 10], fill=ACCENT)
    d.text((64, 60), "My strategic profile", font=_font(56), fill=INK)
    d.text((64, 132), "Measured from how I actually played. No questionnaire.", font=_font(28), fill=MUTED)
    shown = [k for k in ORDER if k in traits][:4]
    y = 210
    for k in shown:
        kind, top = kinds[k]
        v = traits[k]
        name = translate("en", f"trait.{k}.name").replace("\u2013", "-")  # bundled font has no en dash
        d.text((64, y), name, font=_font(34), fill=INK)
        val = display_value(k, v, kind, top)
        d.text((W - 64 - d.textlength(val, font=_font(34)), y), val, font=_font(34), fill=INK)
        frac = 1.0 if v is None else max(0.0, min(1.0, v / top))
        d.rounded_rectangle([64, y + 52, W - 64, y + 62], radius=5, fill=LINE)
        cx = 64 + frac * (W - 128)
        d.ellipse([cx - 11, y + 46, cx + 11, y + 68], fill=ACCENT)
        y += 92
    d.text((64, H - 64), f"{site_name}  ·  How do you really play?", font=_font(26), fill=MUTED)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
