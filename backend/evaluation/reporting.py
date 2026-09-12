from evaluation.runner import RetrievalComparisonReport


_STRATEGY_LABELS = {
    "vector": "Vector (current baseline)",
    "keyword": "Keyword",
    "hybrid": "Hybrid",
    "hybrid_reranked": "Hybrid + Reranking",
}


def print_comparison_report(report: RetrievalComparisonReport) -> None:
    print("Retrieval comparison:")
    for strategy, strategy_metrics in report.strategies.items():
        label = _STRATEGY_LABELS.get(strategy, strategy.title())
        print(f"  {label}:")
        for metrics in strategy_metrics:
            print(
                f"    K={metrics.k}: Hit Rate={metrics.hit_rate_at_k:.1%}, "
                f"Recall={metrics.recall_at_k:.1%}, "
                f"Precision={metrics.precision_at_k:.1%} "
                f"({metrics.evaluated_cases} labeled cases)"
            )
    print_reranking_delta(report)


def print_reranking_delta(report: RetrievalComparisonReport) -> None:
    """Print the actual before/after change from hybrid to hybrid+reranking.

    Only prints when both strategies were measured, and only reports numbers
    actually produced by the evaluation run (never fabricated).
    """
    baseline = report.strategies.get("hybrid")
    reranked = report.strategies.get("hybrid_reranked")
    if baseline is None or reranked is None:
        return

    baseline_by_k = {metrics.k: metrics for metrics in baseline}
    reranked_by_k = {metrics.k: metrics for metrics in reranked}

    print("Hybrid -> Hybrid + Reranking:")
    for k, before in baseline_by_k.items():
        after = reranked_by_k.get(k)
        if after is None:
            continue
        print(f"  K={k}:")
        for name, before_value, after_value in (
            ("Hit Rate", before.hit_rate_at_k, after.hit_rate_at_k),
            ("Recall", before.recall_at_k, after.recall_at_k),
            ("Precision", before.precision_at_k, after.precision_at_k),
        ):
            delta_points = (after_value - before_value) * 100
            print(
                f"    {name}: {before_value:.1%} -> {after_value:.1%} "
                f"(change: {delta_points:+.1f} pp)"
            )