"""Runs infrastructure-free extended evaluation checks (Step 19).

Complements scripts/evaluate_rag.py, which measures retrieval/generation
quality and requires PostgreSQL + OpenAI. This script exercises tool
correctness, escalation correctness, prompt-injection-heuristic behavior,
and multi-turn continuity using in-memory demo data and fakes -- no
external infrastructure required.
"""

from evaluation.extended_checks import run_all_extended_checks


def main() -> int:
    results = run_all_extended_checks()

    print("Extended Evaluation (infrastructure-free checks)")
    print("-------------------------------------------------")
    overall_ok = True
    for result in results:
        status = "PASS" if result.all_passed else "FAIL"
        print(f"{result.name}: {result.passed}/{result.total} {status}")
        for failure in result.failures:
            print(f"  - FAILED: {failure}")
        overall_ok = overall_ok and result.all_passed

    print()
    print(f"Overall: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
