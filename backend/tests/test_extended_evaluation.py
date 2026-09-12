from evaluation.extended_checks import run_all_extended_checks


def test_all_extended_checks_pass() -> None:
    results = run_all_extended_checks()

    for result in results:
        assert result.all_passed, f"{result.name} had failures: {result.failures}"
        assert result.total > 0


def test_extended_checks_cover_all_four_categories() -> None:
    results = run_all_extended_checks()

    names = {result.name for result in results}
    assert names == {
        "tool_correctness",
        "escalation_correctness",
        "prompt_injection_resistance",
        "multiturn_behavior",
    }
