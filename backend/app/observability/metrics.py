"""Metrics abstraction.

A single `MetricsSink` Protocol so call sites never depend on a concrete
backend. The default `LoggingMetricsSink` only writes structured log lines
-- no paid monitoring service is required. Swapping in an
OpenTelemetry-backed sink later means implementing this Protocol once and
injecting it; no call site changes.

Never log: API keys, secrets, credentials, full customer PII, or unredacted
conversation content. Tags passed here should be identifiers (request id,
route, stage name) -- not raw user/customer text.
"""

import logging
from typing import Protocol


logger = logging.getLogger("observability.metrics")


class MetricsSink(Protocol):
    def record_latency(self, stage: str, duration_ms: float, **tags: str) -> None: ...

    def record_value(self, name: str, value: float, **tags: str) -> None: ...

    def increment(self, counter: str, **tags: str) -> None: ...


class LoggingMetricsSink:
    """Default sink: structured logging only, no external dependency."""

    def record_latency(self, stage: str, duration_ms: float, **tags: str) -> None:
        logger.info(
            "latency_ms=%.2f stage=%s %s",
            duration_ms,
            stage,
            tags,
            extra={"stage": stage, "duration_ms": duration_ms, **tags},
        )

    def record_value(self, name: str, value: float, **tags: str) -> None:
        logger.info(
            "metric=%s value=%s %s",
            name,
            value,
            tags,
            extra={"metric": name, "value": value, **tags},
        )

    def increment(self, counter: str, **tags: str) -> None:
        logger.info(
            "counter=%s %s",
            counter,
            tags,
            extra={"counter": counter, **tags},
        )


class NullMetricsSink:
    """No-op sink, useful for tests that don't care about metrics output."""

    def record_latency(self, stage: str, duration_ms: float, **tags: str) -> None:
        return None

    def record_value(self, name: str, value: float, **tags: str) -> None:
        return None

    def increment(self, counter: str, **tags: str) -> None:
        return None
