from __future__ import annotations

import binascii
import math
import random
from dataclasses import dataclass
from typing import Any

from obs_simulator.config import SimulatorConfig
from obs_simulator.utils.parse import clamp_int


@dataclass(frozen=True)
class Targets:
    cpu_target_pct: int
    mem_target_bytes: int
    gpu_util_target_pct: int
    gpu_mem_target_mb: int
    log_rate_lps: int
    db_qps_target: int


def _stable_seed(base_seed: int, name: str) -> int:
    return base_seed + (binascii.crc32(name.encode("utf-8")) & 0xFFFFFFFF)


def _tick_event(name: str, tick: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    e: dict[str, Any] = {"event": name, "tick": tick}
    if extra:
        e.update(extra)
    return e


class Scenario:
    def __init__(self, cfg: SimulatorConfig) -> None:
        self._cfg = cfg
        seed = cfg.global_cfg.seed
        self._rng_cpu = random.Random(_stable_seed(seed, "cpu"))
        self._rng_mem = random.Random(_stable_seed(seed, "mem"))
        self._rng_gpu = random.Random(_stable_seed(seed, "gpu"))
        self._rng_log = random.Random(_stable_seed(seed, "log"))
        self._rng_db = random.Random(_stable_seed(seed, "db"))

        self._prev_step_idx: dict[str, int] = {}
        self._prev_spike_on: dict[str, bool] = {}

    @property
    def interval_seconds(self) -> float:
        return self._cfg.global_cfg.interval_seconds

    def describe_modes(self) -> dict[str, str]:
        g = self._cfg.global_cfg
        cpu_mode = self._cfg.cpu.mode or g.sim_mode
        mem_mode = self._cfg.mem.mode or g.sim_mode
        gpu_mode = self._cfg.gpu.mode or g.sim_mode
        gpu_mem_mode = self._cfg.gpu.mem_mode or gpu_mode
        return {
            "cpu": cpu_mode,
            "mem": mem_mode,
            "gpu": gpu_mode,
            "gpu_mem": gpu_mem_mode,
            "log": self._cfg.log.mode,
            "db": self._cfg.db.mode,
        }

    def _maybe_spike_event(self, key: str, tick: int, is_on: bool, prefix: str) -> list[dict[str, Any]]:
        prev = self._prev_spike_on.get(key)
        self._prev_spike_on[key] = is_on
        if prev is None:
            return []
        if (not prev) and is_on:
            return [_tick_event(f"{prefix}_spike_start", tick)]
        if prev and (not is_on):
            return [_tick_event(f"{prefix}_spike_end", tick)]
        return []

    def _maybe_step_event(self, key: str, tick: int, idx: int, prefix: str, value: int) -> list[dict[str, Any]]:
        prev = self._prev_step_idx.get(key)
        self._prev_step_idx[key] = idx
        if prev is None:
            return []
        if idx != prev:
            return [_tick_event(f"{prefix}_step_change", tick, {"value": value})]
        return []

    def _mode(self, override: str | None) -> str:
        return override or self._cfg.global_cfg.sim_mode

    def _ticks_to_seconds(self, ticks: int) -> float:
        return ticks * self.interval_seconds

    def _wave_int(
        self,
        *,
        key: str,
        tick: int,
        mode: str,
        base: int,
        jitter_amp: int,
        spike_value: int,
        spike_every_ticks: int,
        spike_last_ticks: int,
        step_series: list[int],
        step_every_ticks: int,
        sine_min: int,
        sine_max: int,
        sine_period_ticks: int,
        random_min: int,
        random_max: int,
        outlier_enable: bool,
        outlier_every_ticks: int,
        outlier_value: int,
        clamp_min: int,
        clamp_max: int,
        rng: random.Random,
        prefix: str,
    ) -> tuple[int, list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        value: int

        if mode == "steady":
            value = base
        elif mode == "jitter":
            value = int(round(base + rng.uniform(-jitter_amp, jitter_amp)))
        elif mode == "spike":
            on = (tick % spike_every_ticks) < spike_last_ticks
            events.extend(self._maybe_spike_event(key, tick, on, prefix))
            value = spike_value if on else base
        elif mode == "step":
            if not step_series:
                value = base
            else:
                idx = (tick // step_every_ticks) % len(step_series)
                value = int(step_series[idx])
                events.extend(self._maybe_step_event(key, tick, idx, prefix, value))
        elif mode == "sine":
            t = tick * self.interval_seconds
            period_s = self._ticks_to_seconds(sine_period_ticks)
            if period_s <= 0:
                value = base
            else:
                s = (1.0 + math.sin(2.0 * math.pi * (t / period_s))) / 2.0
                value = int(round(sine_min + (sine_max - sine_min) * s))
        elif mode == "random":
            lo = min(random_min, random_max)
            hi = max(random_min, random_max)
            value = int(rng.randint(lo, hi))
        else:
            value = base

        # Outlier overrides mode target.
        if outlier_enable and tick != 0 and (tick % outlier_every_ticks == 0):
            value = outlier_value
            events.append(_tick_event(f"{prefix}_outlier", tick, {"value": value}))

        value = clamp_int(value, clamp_min, clamp_max)
        return value, events

    def tick(self, tick: int) -> tuple[Targets, list[dict[str, Any]], dict[str, str]]:
        g = self._cfg.global_cfg
        modes = self.describe_modes()
        events: list[dict[str, Any]] = []

        # CPU
        if not self._cfg.cpu.enable:
            cpu_target, cpu_events = 0, []
        else:
            m = modes["cpu"]
            cpu_target, cpu_events = self._wave_int(
                key="cpu",
                tick=tick,
                mode=m,
                base=self._cfg.cpu.target_pct,
                jitter_amp=self._cfg.cpu.jitter_pct,
                spike_value=self._cfg.cpu.spike_pct,
                spike_every_ticks=self._cfg.cpu.spike_every_ticks,
                spike_last_ticks=self._cfg.cpu.spike_last_ticks,
                step_series=self._cfg.cpu.step_series,
                step_every_ticks=self._cfg.cpu.step_every_ticks,
                sine_min=self._cfg.cpu.sine_min,
                sine_max=self._cfg.cpu.sine_max,
                sine_period_ticks=self._cfg.cpu.sine_period_ticks,
                random_min=self._cfg.cpu.random_min,
                random_max=self._cfg.cpu.random_max,
                outlier_enable=self._cfg.cpu.outlier_enable,
                outlier_every_ticks=self._cfg.cpu.outlier_every_ticks,
                outlier_value=self._cfg.cpu.outlier_pct,
                clamp_min=0,
                clamp_max=100,
                rng=self._rng_cpu,
                prefix="cpu",
            )
            events.extend(cpu_events)

        # MEM
        mem_limit = self._cfg.memory_limit_bytes
        mem_allow_over = (modes["mem"] == "oom") or (self._cfg.mem.oom_pct > 100)
        if not self._cfg.mem.enable:
            mem_target, mem_events = 0, []
        else:
            mem_mode = modes["mem"]
            if self._cfg.mem.target_bytes is not None:
                mem_base = self._cfg.mem.target_bytes
            else:
                mem_base = 0
                if mem_limit is not None:
                    mem_base = int(mem_limit * self._cfg.mem.target_pct / 100.0)

            if mem_mode == "leak":
                t = tick * self.interval_seconds
                mem_base = int(mem_base + self._cfg.mem.leak_bytes_per_sec * t)
                mem_mode = "steady"

            if modes["mem"] == "oom":
                if mem_limit is None:
                    raise ValueError("MEM_MODE=oom requires cgroup memory limit")
                mem_target = int(mem_limit * self._cfg.mem.oom_pct / 100.0)
                mem_events = [_tick_event("mem_oom_target", tick, {"value": mem_target})] if tick == 0 else []
            else:
                clamp_max = mem_limit if (mem_limit is not None and not mem_allow_over) else (1 << 62)
                mem_target, mem_events = self._wave_int(
                    key="mem",
                    tick=tick,
                    mode=modes["mem"],
                    base=int(mem_base),
                    jitter_amp=int(self._cfg.mem.jitter_bytes),
                    spike_value=int(mem_base + self._cfg.mem.spike_bytes),
                    spike_every_ticks=self._cfg.mem.spike_every_ticks,
                    spike_last_ticks=self._cfg.mem.spike_last_ticks,
                    step_series=self._cfg.mem.step_series,
                    step_every_ticks=self._cfg.mem.step_every_ticks,
                    sine_min=int(self._cfg.mem.sine_min_bytes),
                    sine_max=int(self._cfg.mem.sine_max_bytes),
                    sine_period_ticks=self._cfg.mem.sine_period_ticks,
                    random_min=int(self._cfg.mem.random_min_bytes),
                    random_max=int(self._cfg.mem.random_max_bytes),
                    outlier_enable=self._cfg.mem.outlier_enable,
                    outlier_every_ticks=self._cfg.mem.outlier_every_ticks,
                    outlier_value=int(self._cfg.mem.outlier_bytes),
                    clamp_min=0,
                    clamp_max=int(clamp_max),
                    rng=self._rng_mem,
                    prefix="mem",
                )
            events.extend(mem_events)

        # GPU (targets only; actual execution is optional at runtime)
        if not self._cfg.gpu.enable:
            gpu_util, gpu_mem, gpu_events = 0, 0, []
        else:
            gpu_util, util_events = self._wave_int(
                key="gpu_util",
                tick=tick,
                mode=modes["gpu"],
                base=self._cfg.gpu.util_target_pct,
                jitter_amp=self._cfg.gpu.util_jitter_pct,
                spike_value=self._cfg.gpu.util_spike_pct,
                spike_every_ticks=self._cfg.gpu.util_spike_every_ticks,
                spike_last_ticks=self._cfg.gpu.util_spike_last_ticks,
                step_series=self._cfg.gpu.util_step_series,
                step_every_ticks=self._cfg.gpu.util_step_every_ticks,
                sine_min=self._cfg.gpu.util_sine_min,
                sine_max=self._cfg.gpu.util_sine_max,
                sine_period_ticks=self._cfg.gpu.util_sine_period_ticks,
                random_min=0,
                random_max=100,
                outlier_enable=self._cfg.gpu.outlier_enable,
                outlier_every_ticks=self._cfg.gpu.outlier_every_ticks,
                outlier_value=self._cfg.gpu.outlier_util_pct,
                clamp_min=0,
                clamp_max=100,
                rng=self._rng_gpu,
                prefix="gpu_util",
            )
            gpu_mem, mem_events = self._wave_int(
                key="gpu_mem",
                tick=tick,
                mode=modes["gpu_mem"],
                base=self._cfg.gpu.mem_target_mb,
                jitter_amp=self._cfg.gpu.mem_jitter_mb,
                spike_value=self._cfg.gpu.mem_target_mb + self._cfg.gpu.mem_spike_mb,
                spike_every_ticks=self._cfg.gpu.mem_spike_every_ticks,
                spike_last_ticks=self._cfg.gpu.mem_spike_last_ticks,
                step_series=self._cfg.gpu.mem_step_series_mb,
                step_every_ticks=self._cfg.gpu.mem_step_every_ticks,
                sine_min=self._cfg.gpu.mem_sine_min_mb,
                sine_max=self._cfg.gpu.mem_sine_max_mb,
                sine_period_ticks=self._cfg.gpu.mem_sine_period_ticks,
                random_min=0,
                random_max=max(0, self._cfg.gpu.mem_sine_max_mb),
                outlier_enable=self._cfg.gpu.outlier_enable and (self._cfg.gpu.outlier_mem_mb > 0),
                outlier_every_ticks=self._cfg.gpu.outlier_every_ticks,
                outlier_value=self._cfg.gpu.outlier_mem_mb,
                clamp_min=0,
                clamp_max=max(0, self._cfg.gpu.mem_sine_max_mb),
                rng=self._rng_gpu,
                prefix="gpu_mem",
            )
            gpu_events = util_events + mem_events
            events.extend(gpu_events)

        # LOG
        if not self._cfg.log.enable:
            log_rate, log_events = 0, []
        else:
            m = self._cfg.log.mode
            base = max(0, int(self._cfg.log.rate))
            if m == "steady":
                log_rate = base
                log_events = []
            elif m in {"jitter", "random"}:
                amp = int(round(base * self._cfg.log.jitter_pct / 100.0))
                log_rate = int(round(base + self._rng_log.uniform(-amp, amp)))
                log_events = []
            elif m == "spike":
                on = (tick % self._cfg.log.burst_every_ticks) < self._cfg.log.burst_last_ticks
                log_events = self._maybe_spike_event("log_burst", tick, on, "log_burst")
                log_rate = self._cfg.log.burst_rate if on else base
            else:
                log_rate, log_events = base, []

            log_rate = max(0, min(20000, int(log_rate)))
            events.extend(log_events)

        # DB QPS target
        db_enabled = self._is_db_enabled()
        if not db_enabled:
            db_qps, db_events = 0, []
        else:
            m = self._cfg.db.mode
            base = max(0, int(self._cfg.db.qps))
            if m == "steady":
                db_qps, db_events = base, []
            elif m == "jitter":
                amp = int(round(base * self._cfg.db.jitter_pct / 100.0))
                db_qps = int(round(base + self._rng_db.uniform(-amp, amp)))
                db_events = []
            elif m == "spike":
                spike_val = self._cfg.db.spike_qps if self._cfg.db.spike_qps is not None else base * 3
                on = (tick % self._cfg.db.spike_every_ticks) < self._cfg.db.spike_last_ticks
                db_events = self._maybe_spike_event("db_qps", tick, on, "db_qps")
                db_qps = int(spike_val if on else base)
            elif m == "step":
                if not self._cfg.db.step_series:
                    db_qps, db_events = base, []
                else:
                    idx = (tick // self._cfg.db.step_every_ticks) % len(self._cfg.db.step_series)
                    db_qps = int(self._cfg.db.step_series[idx])
                    db_events = self._maybe_step_event("db_qps", tick, idx, "db_qps", db_qps)
            elif m == "sine":
                t = tick * self.interval_seconds
                period_s = self._ticks_to_seconds(self._cfg.db.sine_period_ticks)
                if period_s <= 0:
                    db_qps, db_events = base, []
                else:
                    s = (1.0 + math.sin(2.0 * math.pi * (t / period_s))) / 2.0
                    db_qps = int(round(self._cfg.db.sine_min_qps + (self._cfg.db.sine_max_qps - self._cfg.db.sine_min_qps) * s))
                    db_events = []
            elif m == "random":
                lo = min(self._cfg.db.random_min_qps, self._cfg.db.random_max_qps)
                hi = max(self._cfg.db.random_min_qps, self._cfg.db.random_max_qps)
                db_qps = int(self._rng_db.randint(lo, hi))
                db_events = []
            else:
                db_qps, db_events = base, []

            db_qps = max(0, int(db_qps))
            events.extend(db_events)

        targets = Targets(
            cpu_target_pct=int(cpu_target),
            mem_target_bytes=int(mem_target),
            gpu_util_target_pct=int(gpu_util),
            gpu_mem_target_mb=int(gpu_mem),
            log_rate_lps=int(log_rate),
            db_qps_target=int(db_qps),
        )
        return targets, events, modes

    def _is_db_enabled(self) -> bool:
        db = self._cfg.db
        kb = db.kubeblocks
        if db.enable == "false":
            return False
        if db.enable == "true":
            return True
        # auto
        if db.uri:
            return True
        if kb.enable:
            return True
        return False

