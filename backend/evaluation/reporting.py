from evaluation.runner import RetrievalComparisonReport


def print_comparison_report(report: RetrievalComparisonReport) -> None:
    print("Retrieval comparison:")
    for strategy, strategy_metrics in report.strategies.items():
        label = (
            f"{strategy.title()} (current baseline)"
            if strategy == "vector"
            else strategy.title()
        )
        print(f"  {label}:")
        for metrics in strategy_metrics:
            print(
                f"    K={metrics.k}: Hit Rate={metrics.hit_rate_at_k:.1%}, "
                f"Recall={metrics.recall_at_k:.1%}, "
                f"Precision={metrics.precision_at_k:.1%} "
                f"({metrics.evaluated_cases} labeled cases)"
            )