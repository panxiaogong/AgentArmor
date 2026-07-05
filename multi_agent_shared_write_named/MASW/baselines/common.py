"""Shared baseline interfaces and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol


@dataclass(frozen=True)
class BaselinePrediction:
    """Normalized output for all baseline detectors.

    `risk` is a detector-specific score in [0, 1]. It is not assumed to be a
    calibrated probability. `skipped=True` means the baseline could not be run,
    usually because a remote key or optional dependency is missing.
    """

    baseline: str
    predicted_attack: bool
    risk: float
    reason: str
    elapsed_ms: float
    skipped: bool = False
    raw_response: str | None = None


class BaselineDetector(Protocol):
    """Protocol implemented by every baseline detector."""

    name: str

    def predict(self, sample: dict[str, object]) -> BaselinePrediction:
        """Classify one dataset row as attack or benign."""


def now_ns() -> int:
    """Small wrapper used to keep timing logic consistent in baselines."""

    return perf_counter_ns()


def elapsed_ms(started_ns: int) -> float:
    """Convert a perf-counter start timestamp to milliseconds."""

    return (perf_counter_ns() - started_ns) / 1_000_000


def sample_text(sample: dict[str, object]) -> str:
    """Build the text view that external baselines see.

    Tool baselines should not inspect MASW-internal expected labels. They only
    receive the observable user/task/source/content fields.
    """

    return "\n".join(
        [
            f"user_query: {sample.get('user_query', '')}",
            f"task_summary: {sample.get('task_summary', '')}",
            f"source_type: {sample.get('source_type', '')}",
            f"source_uri: {sample.get('source_uri', '')}",
            f"content: {sample.get('content', '')}",
        ]
    )


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)

