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