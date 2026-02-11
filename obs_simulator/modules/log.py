from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from obs_simulator.logfmt import json_line, now_epoch_ms
from obs_simulator.utils.token_bucket import TokenBucket


@dataclass
class LogState:
    tick: int = 0
    scenario: dict[str, str] | None = None
    targets: dict[str, Any] | None = None


class LogModule:
    def __init__(
        self,
        *,
        sim_id: str,
        object_type: str,
        seed: int,
        fmt: str,
        error_pct: int,
        multiline_pct: int,
        max_line_bytes: int,
        extra_fields: dict[str, str],
    ) -> None:
        self._sim_id = sim_id
        self._object_type = object_type
        self._seed = int(seed)
        self._fmt = fmt
        self._error_pct = max(0, min(100, int(error_pct)))
        self._multiline_pct = max(0, min(100, int(multiline_pct)))
        self._max_line_bytes = max(0, int(max_line_bytes))
        self._extra_fields = dict(extra_fields)

        self._stop = threading.Event()
        self._bucket = TokenBucket(rate_per_sec=0.0, capacity=1.0)
        self._lock = threading.Lock()
        self._state = LogState()
        self._lines_total = 0
        self._lines_since_last = 0

        self._thread = threading.Thread(target=self._run, name="log-module", daemon=True)

    def start(self, initial_rate_lps: int) -> None:
        self.set_rate_lps(initial_rate_lps)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._bucket.set_rate(0.0)
        self._thread.join(timeout=5.0)

    def set_rate_lps(self, rate_lps: int) -> None:
        rate = max(0.0, float(rate_lps))
        cap = max(1.0, rate) if rate > 0 else 1.0
        self._bucket.set_rate(rate, capacity=cap)

    def update_state(self, *, tick: int, scenario: dict[str, str], targets: dict[str, Any]) -> None:
        with self._lock:
            self._state.tick = int(tick)
            self._state.scenario = dict(scenario)
            self._state.targets = dict(targets)

    def snapshot_metrics(self) -> dict[str, int]:
        with self._lock:
            out = {
                "log_lines_total": int(self._lines_total),
                "log_lines_last_interval": int(self._lines_since_last),
            }
            self._lines_since_last = 0
            return out

    def _emit(self, record: dict[str, Any]) -> None:
        if self._fmt == "text":
            print(record.get("msg", ""), flush=True)
            return
        json_line(record)

    def _make_msg(self, level: str) -> str:
        if level != "ERROR" or self._max_line_bytes <= 0:
            return "obs-simulator log"
        # Produce a large msg string to test truncation. Use ASCII to keep byte count stable.
        size = min(self._max_line_bytes, 256 * 1024)
        return "X" * size

    def _run(self) -> None:
        rng = __import__("random").Random(self._seed)
        while not self._stop.is_set():
            ok = self._bucket.acquire(1.0, stop_event=self._stop)
            if not ok:
                break

            with self._lock:
                tick = self._state.tick
                scenario = self._state.scenario or {}
                targets = self._state.targets or {}

            is_error = rng.randint(1, 100) <= self._error_pct
            level = "ERROR" if is_error else "INFO"
            msg = self._make_msg(level)

            multiline = rng.randint(1, 100) <= self._multiline_pct
            if multiline:
                event_id = str(uuid.uuid4())
                lines = 3
                for i in range(lines):
                    record = {
                        "kind": "log",
                        "ts_ms": now_epoch_ms(),
                        "level": level,
                        "sim_id": self._sim_id,
                        "object_type": self._object_type,
                        "tick": tick,
                        "scenario": scenario,
                        "targets": targets,
                        "event_id": event_id,
                        "event_line": i + 1,
                        "event_lines": lines,
                        "msg": msg,
                        **self._extra_fields,
                    }
                    self._emit(record)
                    with self._lock:
                        self._lines_total += 1
                        self._lines_since_last += 1
            else:
                record = {
                    "kind": "log",
                    "ts_ms": now_epoch_ms(),
                    "level": level,
                    "sim_id": self._sim_id,
                    "object_type": self._object_type,
                    "tick": tick,
                    "scenario": scenario,
                    "targets": targets,
                    "msg": msg,
                    **self._extra_fields,
                }
                self._emit(record)
                with self._lock:
                    self._lines_total += 1
                    self._lines_since_last += 1

            # A tiny sleep keeps the scheduler fair at very high rates.
            if self._bucket is not None:
                time.sleep(0)
