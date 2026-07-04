"""Robust parsing helpers for coercing free-form LLM output into numbers/JSON.

LLMs don't always honour "reply with only a number/JSON" instructions, so these
helpers extract the intended value defensively instead of trusting the format.
"""

from __future__ import annotations

import json
import re

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_score(text: str, default: float = 0.0) -> float:
    """Extract the first number from ``text`` and clamp it to [0, 1]."""
    match = _FLOAT_RE.search(text or "")
    if not match:
        return default
    try:
        value = float(match.group())
    except ValueError:
        return default
    return max(0.0, min(1.0, value))


def parse_json_object(text: str) -> dict | None:
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
