"""Stage-latency measurement.

Used to time retrieval, reranking, LLM, and tool calls independently so
total request latency can be broken down by stage (see docs/observability.md).
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage: str
    duration_ms: float


@dataclass(slots=True)
class RequestTimings:
    stages: list[StageTiming] = field(default_factory=list)

    def add(self, stage: str, duration_ms: float) -> None:
        self.stages.append(StageTiming(stage, duration_ms))

    @property
    def total_ms(self) -> float:
        return sum(stage.duration_ms for stage in self.stages)

    def as_dict(self) -> dict[str, float]:
        return {stage.stage: stage.duration_ms for stage in self.stages}


@contextmanager
def measure(timings: RequestTimings, stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings.add(stage, (time.perf_counter() - start) * 1000)
