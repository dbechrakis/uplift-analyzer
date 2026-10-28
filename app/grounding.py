from __future__ import annotations

import re
from typing import Any

NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
ALLOWED_CONSTANTS = (0.001, 0.01, 0.05, 0.1)


def _flatten(value: Any, out: list[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        out.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            _flatten(item, out)
    elif isinstance(value, list | tuple):
        for item in value:
            _flatten(item, out)


def known_values(*sources: Any) -> list[float]:
    values: list[float] = []
    for source in sources:
        _flatten(source, values)
    forms: list[float] = []
    for v in values:
        forms.extend((v, v * 100, abs(v), abs(v) * 100))
    return forms


def _decimals(token: str) -> int:
    mantissa = re.split(r"[eE]", token)[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def _matches(number: float, token: str, forms: list[float]) -> bool:
    if "e" in token.lower():
        return any(f != 0 and abs(number - f) <= 0.01 * abs(f) for f in forms)
    tolerance = 0.5 * 10 ** -_decimals(token) + 1e-9
    return any(abs(number - f) <= tolerance for f in forms)


def ungrounded_numbers(text: str, *sources: Any) -> list[str]:
    forms = known_values(*sources)
    flagged: list[str] = []
    for match in NUMBER.finditer(text):
        token = match.group().rstrip(",")
        try:
            number = float(token.replace(",", ""))
        except ValueError:
            continue
        if number in ALLOWED_CONSTANTS or abs(number) in ALLOWED_CONSTANTS:
            continue
        if not _matches(abs(number), token.lstrip("+-"), forms) and token not in flagged:
            flagged.append(token)
    return flagged
