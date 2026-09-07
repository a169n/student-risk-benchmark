from __future__ import annotations

from collections.abc import Sequence
from typing import SupportsFloat

import math


def clamp(value: SupportsFloat, low: float, high: float) -> float:
    return max(low, min(float(value), high))


def clip01(value: SupportsFloat) -> float:
    return clamp(value, 0.0, 1.0)


def safe_mean(values: Sequence[float]) -> float | None:
    cleaned = [value for value in values if value is not None and not math.isnan(value)]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def recent_trend(values: Sequence[float], *, scale: float) -> float:
    cleaned = [value for value in values if value is not None and not math.isnan(value)]
    if len(cleaned) < 2:
        return 0.0
    window = cleaned[-3:]
    return clamp((window[-1] - window[0]) / scale, -1.0, 1.0)


def normalized_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return clip01(numerator / denominator)


def stable_probability_weights(weights: dict[str, float]) -> tuple[list[str], list[float]]:
    keys = sorted(weights)
    total = sum(weights[key] for key in keys)
    return keys, [weights[key] / total for key in keys]
