from app.api.rate_limit import InMemoryRateLimiter


def test_allows_requests_within_the_limit() -> None:
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)

    assert limiter.allow("client-a", now=0.0) is True
    assert limiter.allow("client-a", now=1.0) is True
    assert limiter.allow("client-a", now=2.0) is True


def test_rejects_requests_beyond_the_limit_within_the_window() -> None:
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)

    assert limiter.allow("client-a", now=0.0) is True
    assert limiter.allow("client-a", now=1.0) is True
    assert limiter.allow("client-a", now=2.0) is False


def test_window_expiry_allows_requests_again() -> None:
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=10)

    assert limiter.allow("client-a", now=0.0) is True
    assert limiter.allow("client-a", now=5.0) is False
    assert limiter.allow("client-a", now=11.0) is True


def test_different_clients_have_independent_limits() -> None:
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)

    assert limiter.allow("client-a", now=0.0) is True
    assert limiter.allow("client-b", now=0.0) is True
    assert limiter.allow("client-a", now=1.0) is False


def test_reset_clears_a_clients_history() -> None:
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    limiter.allow("client-a", now=0.0)

    limiter.reset("client-a")

    assert limiter.allow("client-a", now=0.1) is True


def test_rejects_non_positive_configuration() -> None:
    import pytest

    with pytest.raises(ValueError):
        InMemoryRateLimiter(max_requests=0, window_seconds=60)
    with pytest.raises(ValueError):
        InMemoryRateLimiter(max_requests=1, window_seconds=0)
