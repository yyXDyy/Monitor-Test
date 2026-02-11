from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float | None = None) -> None:
        self._cond = threading.Condition()
        self._rate = max(0.0, float(rate_per_sec))
        self._capacity = float(capacity) if capacity is not None else max(1.0, self._rate)
        self._tokens = self._capacity
        self._last = time.monotonic()

    def set_rate(self, rate_per_sec: float, capacity: float | None = None) -> None:
        with self._cond:
            self._refill_locked()
            self._rate = max(0.0, float(rate_per_sec))
            if capacity is not None:
                self._capacity = max(1.0, float(capacity))
            self._tokens = min(self._tokens, self._capacity)
            self._cond.notify_all()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last = now

    def acquire(self, tokens: float = 1.0, stop_event: threading.Event | None = None) -> bool:
        need = float(tokens)
        with self._cond:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return False
                self._refill_locked()
                if self._tokens >= need:
                    self._tokens -= need
                    return True

                if self._rate <= 0:
                    self._cond.wait(timeout=0.5)
                    continue

                missing = need - self._tokens
                wait_s = missing / self._rate
                self._cond.wait(timeout=min(wait_s, 0.5))
