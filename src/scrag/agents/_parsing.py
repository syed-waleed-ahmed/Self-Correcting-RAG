"""Robust parsing helpers for coercing free-form LLM output into numbers/JSON.

LLMs don't always honour "reply with only a number/JSON" instructions, so these
helpers extract the intended value defensively instead of trusting the format.
"""

from __future__ import annotations

import json
import re

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
# A number, optionally followed by a percent sign or a "/ N" / "out of N" scale.
_SCORE_RE = re.compile(
    rf"(?P<num>{_NUMBER})\s*(?:(?P<pct>%)|(?:/|\bout\s+of\b)\s*(?P<den>{_NUMBER}))?",
    re.IGNORECASE,
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def normalize_score(text: str | None) -> float | None:
    """Parse the first score in ``text`` into ``[0, 1]``, or ``None`` if there is none.

    Understands decimals (``0.8``, ``.8``), percentages (``85%``) and explicit
    scales (``8/10``, ``7.5 / 10``, ``9 out of 10``). A bare number outside
    ``[0, 1]`` is clamped, because without a stated scale it cannot be
    reinterpreted safely. A zero or negative denominator yields ``None``.
    """
    match = _SCORE_RE.search(text or "")
    if not match:
        return None
    try:
        value = float(match.group("num"))
        if match.group("pct"):
            value /= 100.0
        elif match.group("den") is not None:
            denominator = float(match.group("den"))
            if denominator <= 0:
                return None
            value /= denominator
    except ValueError:
        return None
    return max(0.0, min(1.0, value))


def parse_score(text: str | None, default: float = 0.0) -> float:
    """Like :func:`normalize_score`, but returns ``default`` when no score is found."""
    value = normalize_score(text)
    return default if value is None else value


def parse_json_object(text: str | None) -> dict | None:
    """Parse a JSON object from ``text``, tolerating surrounding prose/fences."""
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
