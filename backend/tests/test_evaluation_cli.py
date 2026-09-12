from pytest import CaptureFixture

from evaluation.metrics import RetrievalMetrics
from evaluation.reporting import print_comparison_report
from evaluation.runner import RetrievalComparisonReport


def test_prints_separate_strategy_metrics_and_baseline_label(
    capsys: CaptureFixture[str],
) -> None:
    metric = RetrievalMetrics(
        k=5,
        evaluated_cases=5,
        recall_at_k=0.8,
        precision_at_k=0.4,
        hit_rate_at_k=1.0,
    )
    report = RetrievalComparisonReport(
        dataset_size=10,
        strategies={
            "vector": (metric,),
            "keyword": (metric,),
            "hybrid": (metric,),
        },
    )

    print_comparison_report(report)

    output = capsys.readouterr().out
    assert "Vector (current baseline)" in output
    assert "Keyword" in output
    assert "Hybrid" in output
    assert output.count("Hit Rate=100.0%") == 3


def test_prints_actual_before_after_delta_for_hybrid_reranking(
    capsys: CaptureFixture[str],
) -> None:
    before = RetrievalMetrics(
        k=5,
        evaluated_cases=5,
        recall_at_k=0.6,
        precision_at_k=0.4,
        hit_rate_at_k=0.8,
    )
    after = RetrievalMetrics(
        k=5,
        evaluated_cases=5,
        recall_at_k=0.8,
        precision_at_k=0.4,
        hit_rate_at_k=1.0,
    )
    report = RetrievalComparisonReport(
        dataset_size=10,
        strategies={"hybrid": (before,), "hybrid_reranked": (after,)},
    )

    print_comparison_report(report)

    output = capsys.readouterr().out
    assert "Hybrid + Reranking" in output
    assert "Hybrid -> Hybrid + Reranking" in output
    assert "Recall: 60.0% -> 80.0% (change: +20.0 pp)" in output


def test_no_delta_section_when_reranking_not_measured(
    capsys: CaptureFixture[str],
) -> None:
    metric = RetrievalMetrics(
        k=5,
        evaluated_cases=5,
        recall_at_k=0.8,
        precision_at_k=0.4,
        hit_rate_at_k=1.0,
    )
    report = RetrievalComparisonReport(dataset_size=10, strategies={"hybrid": (metric,)})

    print_comparison_report(report)

    output = capsys.readouterr().out
    assert "Hybrid -> Hybrid + Reranking" not in output