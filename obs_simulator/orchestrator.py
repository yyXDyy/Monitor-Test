from __future__ import annotations

import os
import signal
import time

from obs_simulator.config import ParseError, SimulatorConfig, load_config
from obs_simulator.logfmt import json_line, now_epoch_ms
from obs_simulator.runtime import RuntimeManager
from obs_simulator.scenario import Scenario
from obs_simulator.utils.cgroup import get_cpu_usage_usec


def _total_cpu_seconds() -> float:
    t = os.times()
    return float(t.user + t.system + t.children_user + t.children_system)


def _cpu_pct_total(prev_cpu_s: float, prev_wall_s: float) -> tuple[float, float, float]:
    wall = time.monotonic()
    cpu = _total_cpu_seconds()
    dt_wall = wall - prev_wall_s
    dt_cpu = cpu - prev_cpu_s
    if dt_wall <= 0:
        return 0.0, cpu, wall
    cores = max(1, os.cpu_count() or 1)
    pct = (dt_cpu / (dt_wall * cores)) * 100.0
    return max(0.0, pct), cpu, wall


def _cpu_from_cgroup(
    prev_usage_usec: int | None, prev_wall_s: float
) -> tuple[float | None, float | None, int | None, float]:
    """
    Returns:
        (cpu_cores_used, cpu_pct_total, usage_usec, wall_s)
    """
    usage = get_cpu_usage_usec()
    if usage is None or prev_usage_usec is None:
        return None, None, prev_usage_usec, prev_wall_s
    wall = time.monotonic()
    dt_wall = wall - prev_wall_s
    dt_usec = usage - prev_usage_usec
    if dt_wall <= 0 or dt_usec < 0:
        return None, None, usage, wall
    cores_used = float(dt_usec) / (dt_wall * 1_000_000.0)
    cores = max(1, os.cpu_count() or 1)
    pct_total = (cores_used / cores) * 100.0
    return cores_used, max(0.0, pct_total), usage, wall


def _log_boot(cfg: SimulatorConfig) -> None:
    json_line({"kind": "config", "ts_ms": now_epoch_ms(), "config": cfg.to_effective_dict()})


def _event_sink(cfg: SimulatorConfig):
    def _sink(ev: dict) -> None:
        if not cfg.global_cfg.event_log_enable:
            return
        json_line(
            {
                "kind": "event",
                "ts_ms": now_epoch_ms(),
                "sim_id": cfg.global_cfg.sim_id,
                "object_type": cfg.global_cfg.object_type,
                **ev,
            }
        )

    return _sink


def _error_sink(cfg: SimulatorConfig):
    def _sink(msg: str) -> None:
        json_line(
            {
                "kind": "error",
                "ts_ms": now_epoch_ms(),
                "sim_id": cfg.global_cfg.sim_id,
                "error": msg,
            }
        )

    return _sink


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        cfg = load_config()
    except ParseError as e:
        json_line({"kind": "error", "ts_ms": now_epoch_ms(), "error": str(e)})
        return 2

    _log_boot(cfg)

    stop = {"flag": False}

    def _handle_sig(signum: int, _frame) -> None:
        stop["flag"] = True
        json_line({"kind": "signal", "ts_ms": now_epoch_ms(), "signal": int(signum)})

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    scenario = Scenario(cfg)

    runtime = RuntimeManager(
        cfg=cfg,
        event_sink=_event_sink(cfg),
        error_sink=_error_sink(cfg),
    )

    try:
        runtime.start()

        start_wall = time.monotonic()
        tick = 0
        next_tick = time.monotonic()
        prev_cpu = _total_cpu_seconds()
        prev_wall_times = time.monotonic()
        prev_cg = get_cpu_usage_usec()
        prev_wall_cg = prev_wall_times

        while not stop["flag"]:
            if cfg.global_cfg.duration_seconds > 0:
                if (time.monotonic() - start_wall) >= cfg.global_cfg.duration_seconds:
                    break

            targets, events, modes = scenario.tick(tick)

            runtime.apply_targets(targets, tick, modes)

            if cfg.global_cfg.event_log_enable:
                for ev in events:
                    json_line(
                        {
                            "kind": "event",
                            "ts_ms": now_epoch_ms(),
                            "sim_id": cfg.global_cfg.sim_id,
                            "object_type": cfg.global_cfg.object_type,
                            **ev,
                        }
                    )

            cpu_cores_used, cpu_pct_total, prev_cg, prev_wall_cg = _cpu_from_cgroup(prev_cg, prev_wall_cg)
            if cpu_pct_total is None:
                cpu_pct_total, prev_cpu, prev_wall_times = _cpu_pct_total(prev_cpu, prev_wall_times)
            else:
                # Keep times-based wall in sync for consistency in logs.
                prev_wall_times = prev_wall_cg

            actuals: dict[str, object] = {
                "cpu_pct_total": round(cpu_pct_total, 3),
            }
            if cpu_cores_used is not None:
                actuals["cpu_cores_used"] = round(cpu_cores_used, 4)
            actuals.update(runtime.collect_metrics())

            if cfg.global_cfg.heartbeat_enable:
                json_line(
                    {
                        "kind": "heartbeat",
                        "ts_ms": now_epoch_ms(),
                        "sim_id": cfg.global_cfg.sim_id,
                        "object_type": cfg.global_cfg.object_type,
                        "tick": tick,
                        "scenario": modes,
                        "targets": {
                            "cpu_pct": targets.cpu_target_pct,
                            "mem_bytes": targets.mem_target_bytes,
                            "gpu_util_pct": targets.gpu_util_target_pct,
                            "gpu_mem_mb": targets.gpu_mem_target_mb,
                            "log_lps": targets.log_rate_lps,
                            "db_qps": targets.db_qps_target,
                        },
                        "actuals": actuals,
                    }
                )

            tick += 1
            next_tick += cfg.global_cfg.interval_seconds
            sleep_s = max(0.0, next_tick - time.monotonic())
            time.sleep(sleep_s)

    finally:
        runtime.stop()

    json_line({"kind": "shutdown", "ts_ms": now_epoch_ms(), "sim_id": cfg.global_cfg.sim_id})
    return 0
