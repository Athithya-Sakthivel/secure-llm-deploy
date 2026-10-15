"""Bounded in-process sliding-window rate limiter."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict, deque
from time import monotonic


class SlidingWindowRateLimiter:
    def __init__(
        self,
        *,
        max_requests: int = 5,
        window_seconds: int = 60,
        max_clients: int = 10000,
    ) -> None:
        if min(max_requests, window_seconds, max_clients) < 1:
            raise ValueError("Rate-limit parameters must be positive")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def allow(self, client_key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = monotonic()
        cutoff = now - self.window_seconds

        async with self._lock:
            events = self._events.get(client_key)

            if events is not None:
                while events and events[0] <= cutoff:
                    events.popleft()

                if not events:
                    self._events.pop(client_key, None)
                    events = None

            if events is None:
                if len(self._events) >= self.max_clients:
                    expired = [
                        key
                        for key, values in self._events.items()
                        if not values or values[-1] <= cutoff
                    ]
                    for key in expired:
                        self._events.pop(key, None)

                if len(self._events) >= self.max_clients:
                    return False, self.window_seconds

                events = deque()
                self._events[client_key] = events

            self._events.move_to_end(client_key)

            if len(events) >= self.max_requests:
                retry_after = max(
                    1,
                    math.ceil(self.window_seconds - (now - events[0])),
                )
                return False, retry_after

            events.append(now)
            return True, 0