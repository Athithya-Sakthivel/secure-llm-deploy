"""Unit tests for the sliding-window rate limiter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import secure_llm_deploy.rate_limit as rate_limit_module
from secure_llm_deploy.rate_limit import SlidingWindowRateLimiter


@dataclass
class ManualClock:
    now: float = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Clock cannot move backwards")
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> ManualClock:
    manual_clock = ManualClock()
    monkeypatch.setattr(rate_limit_module, "monotonic", manual_clock)
    return manual_clock


@pytest.fixture
def limiter(clock: ManualClock) -> SlidingWindowRateLimiter:
    del clock
    return SlidingWindowRateLimiter(
        max_requests=5,
        window_seconds=60,
        max_clients=100,
    )


@pytest.mark.asyncio
async def test_first_five_requests_are_allowed(
    limiter: SlidingWindowRateLimiter,
) -> None:
    for _ in range(5):
        allowed, retry_after = await limiter.allow("client-a")

        assert allowed
        assert retry_after == 0


@pytest.mark.asyncio
async def test_sixth_request_is_rejected(
    limiter: SlidingWindowRateLimiter,
) -> None:
    for _ in range(5):
        await limiter.allow("client-a")

    allowed, retry_after = await limiter.allow("client-a")

    assert not allowed
    assert retry_after == 60


@pytest.mark.asyncio
async def test_retry_after_decreases_as_window_expires(
    limiter: SlidingWindowRateLimiter,
    clock: ManualClock,
) -> None:
    for _ in range(5):
        await limiter.allow("client-a")

    clock.advance(11.2)
    allowed, retry_after = await limiter.allow("client-a")

    assert not allowed
    assert retry_after == 49


@pytest.mark.asyncio
async def test_client_identities_have_separate_windows(
    limiter: SlidingWindowRateLimiter,
) -> None:
    for _ in range(5):
        allowed, _ = await limiter.allow("client-a")
        assert allowed

    blocked, _ = await limiter.allow("client-a")
    other_allowed, other_retry_after = await limiter.allow("client-b")

    assert not blocked
    assert other_allowed
    assert other_retry_after == 0


@pytest.mark.asyncio
async def test_requests_are_allowed_after_window_expiry(
    limiter: SlidingWindowRateLimiter,
    clock: ManualClock,
) -> None:
    for _ in range(5):
        await limiter.allow("client-a")

    clock.advance(60)

    allowed, retry_after = await limiter.allow("client-a")

    assert allowed
    assert retry_after == 0


@pytest.mark.asyncio
async def test_max_client_capacity_rejects_new_identities(
    clock: ManualClock,
) -> None:
    limiter = SlidingWindowRateLimiter(
        max_requests=2,
        window_seconds=10,
        max_clients=2,
    )

    assert await limiter.allow("client-a") == (True, 0)
    assert await limiter.allow("client-b") == (True, 0)

    allowed, retry_after = await limiter.allow("client-c")

    assert not allowed
    assert retry_after == 10


@pytest.mark.asyncio
async def test_active_clients_are_not_evicted_at_capacity(
    clock: ManualClock,
) -> None:
    limiter = SlidingWindowRateLimiter(
        max_requests=2,
        window_seconds=10,
        max_clients=2,
    )

    assert await limiter.allow("client-a") == (True, 0)
    assert await limiter.allow("client-b") == (True, 0)

    # A new identity cannot evict either active client.
    assert await limiter.allow("client-c") == (False, 10)

    # An existing identity can still use its remaining allowance.
    assert await limiter.allow("client-a") == (True, 0)


@pytest.mark.asyncio
async def test_expired_clients_are_removed_when_capacity_is_needed(
    clock: ManualClock,
) -> None:
    limiter = SlidingWindowRateLimiter(
        max_requests=2,
        window_seconds=10,
        max_clients=2,
    )

    assert await limiter.allow("client-a") == (True, 0)
    assert await limiter.allow("client-b") == (True, 0)

    clock.advance(10)

    # The old client records have expired, so this identity can enter.
    assert await limiter.allow("client-c") == (True, 0)


@pytest.mark.asyncio
async def test_invalid_constructor_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        SlidingWindowRateLimiter(max_requests=0)

    with pytest.raises(ValueError, match="must be positive"):
        SlidingWindowRateLimiter(window_seconds=0)

    with pytest.raises(ValueError, match="must be positive"):
        SlidingWindowRateLimiter(max_clients=0)


@pytest.mark.asyncio
async def test_many_distinct_clients_do_not_share_request_counters(
    limiter: SlidingWindowRateLimiter,
) -> None:
    for client_number in range(20):
        allowed, retry_after = await limiter.allow(
            f"client-{client_number}"
        )
        assert allowed
        assert retry_after == 0