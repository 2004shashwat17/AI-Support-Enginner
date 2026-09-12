import asyncio

from app.observability.context import (
    correlation_tags,
    get_conversation_id,
    get_request_id,
    set_conversation_id,
    set_request_id,
)
from app.observability.cost import TokenUsage, estimate_cost
from app.observability.metrics import LoggingMetricsSink, NullMetricsSink
from app.observability.timing import RequestTimings, measure


def test_request_and_conversation_ids_round_trip() -> None:
    set_request_id("req-1")
    set_conversation_id("conv-1")

    assert get_request_id() == "req-1"
    assert get_conversation_id() == "conv-1"
    assert correlation_tags() == {"request_id": "req-1", "conversation_id": "conv-1"}

    set_request_id(None)
    set_conversation_id(None)


def test_correlation_tags_omit_unset_ids() -> None:
    set_request_id(None)
    set_conversation_id(None)

    assert correlation_tags() == {}


def test_measure_records_a_stage_duration() -> None:
    timings = RequestTimings()

    with measure(timings, "retrieval"):
        pass

    assert len(timings.stages) == 1
    assert timings.stages[0].stage == "retrieval"
    assert timings.stages[0].duration_ms >= 0


def test_total_ms_sums_all_stages() -> None:
    timings = RequestTimings()
    timings.add("retrieval", 10.0)
    timings.add("generation", 20.0)

    assert timings.total_ms == 30.0
    assert timings.as_dict() == {"retrieval": 10.0, "generation": 20.0}


def test_measure_records_duration_even_when_the_block_raises() -> None:
    timings = RequestTimings()

    try:
        with measure(timings, "retrieval"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert len(timings.stages) == 1


def test_null_metrics_sink_does_nothing() -> None:
    sink = NullMetricsSink()
    sink.record_latency("stage", 1.0)
    sink.record_value("metric", 1.0)
    sink.increment("counter")


def test_logging_metrics_sink_does_not_raise(caplog) -> None:
    sink = LoggingMetricsSink()
    with caplog.at_level("INFO"):
        sink.record_latency("retrieval", 12.5, request_id="req-1")
        sink.record_value("llm_prompt_tokens", 42, request_id="req-1")
        sink.increment("escalation_created", reason="user_requested_human")

    assert any("latency_ms" in message for message in caplog.messages)


# Cost: never fabricated when pricing is unconfigured.
def test_cost_is_unknown_without_configured_pricing() -> None:
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50)

    estimate = estimate_cost(usage, prompt_price_per_1k=None, completion_price_per_1k=None)

    assert estimate.prompt_cost_usd is None
    assert estimate.completion_cost_usd is None
    assert estimate.total_cost_usd is None


def test_cost_is_computed_when_pricing_is_configured() -> None:
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)

    estimate = estimate_cost(
        usage, prompt_price_per_1k=0.01, completion_price_per_1k=0.03
    )

    assert estimate.prompt_cost_usd == 0.01
    assert estimate.completion_cost_usd == 0.015
    assert estimate.total_cost_usd == 0.025


def test_partial_pricing_still_reports_unknown_total() -> None:
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)

    estimate = estimate_cost(usage, prompt_price_per_1k=0.01, completion_price_per_1k=None)

    assert estimate.prompt_cost_usd == 0.01
    assert estimate.completion_cost_usd is None
    assert estimate.total_cost_usd is None
