from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from obs_simulator.config import SimulatorConfig
from obs_simulator.modules.cpu import CPUModule, resolve_worker_count
from obs_simulator.modules.db import DBModule
from obs_simulator.modules.gpu import GPUModule
from obs_simulator.modules.log import LogModule
from obs_simulator.modules.memory import MemoryModule
from obs_simulator.scenario import Targets


@dataclass
class ManagedModule:
    name: str
    apply_targets: Callable[[Targets, int, dict[str, str]], None]
    snapshot_metrics: Callable[[], dict[str, object]]
    stop: Callable[[], None]


def should_start_cpu(cfg: SimulatorConfig) -> bool:
    if not cfg.cpu.enable:
        return False
    if not cfg.global_cfg.strict_idle_mode:
        return True

    mode = cfg.cpu.mode or cfg.global_cfg.sim_mode
    if mode != "steady":
        return True
    if cfg.cpu.target_pct > 0:
        return True
    if cfg.cpu.outlier_enable:
        return True
    return False


def should_start_mem(cfg: SimulatorConfig) -> bool:
    if not cfg.mem.enable:
        return False
    if not cfg.global_cfg.strict_idle_mode:
        return True

    mode = cfg.mem.mode or cfg.global_cfg.sim_mode
    if mode != "steady":
        return True
    if (cfg.mem.target_bytes or 0) > 0:
        return True
    if cfg.mem.target_pct > 0:
        return True
    if cfg.mem.outlier_enable:
        return True
    if cfg.mem.leak_bytes_per_sec > 0:
        return True
    return False


def should_start_log(cfg: SimulatorConfig) -> bool:
    return cfg.log.enable


def should_start_gpu(cfg: SimulatorConfig) -> bool:
    return cfg.gpu.enable


def should_start_db(cfg: SimulatorConfig) -> bool:
    return cfg.db.enable != "false" and (cfg.db.uri or cfg.db.kubeblocks.enable or cfg.db.enable == "true")


class RuntimeManager:
    def __init__(
        self,
        *,
        cfg: SimulatorConfig,
        event_sink: Callable[[dict], None],
        error_sink: Callable[[str], None],
    ) -> None:
        self._cfg = cfg
        self._event_sink = event_sink
        self._error_sink = error_sink
        self._modules: list[ManagedModule] = []

    def start(self) -> None:
        cfg = self._cfg

        if should_start_cpu(cfg):
            cpu_workers = resolve_worker_count(cfg.cpu.cores)
            cpu_mod = CPUModule.start(
                workers=cpu_workers,
                cycle_ms=cfg.cpu.cycle_ms,
                initial_target_pct=0,
            )
            self._modules.append(
                ManagedModule(
                    name="cpu",
                    apply_targets=lambda targets, _tick, _modes: cpu_mod.set_target_pct(targets.cpu_target_pct),
                    snapshot_metrics=lambda: {},
                    stop=cpu_mod.close,
                )
            )

        if should_start_mem(cfg):
            mem_mod = MemoryModule.start(
                initial_target_bytes=0,
                touch=cfg.mem.touch,
                release_smooth=cfg.mem.release_smooth,
            )
            self._modules.append(
                ManagedModule(
                    name="mem",
                    apply_targets=lambda targets, _tick, _modes: mem_mod.set_target_bytes(targets.mem_target_bytes),
                    snapshot_metrics=lambda: mem_mod.metrics(),
                    stop=mem_mod.close,
                )
            )

        if should_start_log(cfg):
            import binascii

            log_mod = LogModule(
                sim_id=cfg.global_cfg.sim_id,
                object_type=cfg.global_cfg.object_type,
                seed=cfg.global_cfg.seed + (binascii.crc32(b"log") & 0xFFFFFFFF),
                fmt=cfg.log.fmt,
                error_pct=cfg.log.error_pct,
                multiline_pct=cfg.log.multiline_pct,
                max_line_bytes=cfg.log.max_line_bytes,
                extra_fields=cfg.log.fields_extra,
            )
            log_mod.start(initial_rate_lps=cfg.log.rate)
            self._modules.append(
                ManagedModule(
                    name="log",
                    apply_targets=lambda targets, tick, modes: self._apply_log(log_mod, targets, tick, modes),
                    snapshot_metrics=lambda: log_mod.snapshot_metrics(),
                    stop=log_mod.stop,
                )
            )

        if should_start_gpu(cfg):
            gpu_mod = GPUModule.start(
                gpu_device=cfg.gpu.device,
                util_duty_cycle_ms=cfg.gpu.util_duty_cycle_ms,
                mem_free_policy=cfg.gpu.mem_free_policy,
                initial_util_pct=0,
                initial_mem_mb=0,
                event_sink=self._event_sink,
            )
            if gpu_mod is not None:
                self._modules.append(
                    ManagedModule(
                        name="gpu",
                        apply_targets=lambda targets, _tick, _modes: self._apply_gpu(gpu_mod, targets),
                        snapshot_metrics=lambda: {},
                        stop=gpu_mod.close,
                    )
                )

        if should_start_db(cfg):
            db_mod = DBModule(
                cfg=cfg.db,
                sim_id=cfg.global_cfg.sim_id,
                interval_seconds=cfg.global_cfg.interval_seconds,
                event_sink=self._event_sink,
            )
            try:
                db_mod.start()
            except Exception as e:
                self._error_sink(f"db_start_failed: {e}")
                if cfg.db.enable == "true":
                    raise
            else:
                self._modules.append(
                    ManagedModule(
                        name="db",
                        apply_targets=lambda targets, tick, _modes: self._apply_db(db_mod, targets, tick),
                        snapshot_metrics=lambda: self._snapshot_db(db_mod),
                        stop=db_mod.stop,
                    )
                )

    def apply_targets(self, targets: Targets, tick: int, modes: dict[str, str]) -> None:
        for module in self._modules:
            module.apply_targets(targets, tick, modes)

    def collect_metrics(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for module in self._modules:
            out.update(module.snapshot_metrics())
        return out

    def stop(self) -> None:
        for module in reversed(self._modules):
            try:
                module.stop()
            except Exception:
                continue

    @staticmethod
    def _apply_log(log_mod: LogModule, targets: Targets, tick: int, modes: dict[str, str]) -> None:
        log_mod.set_rate_lps(targets.log_rate_lps)
        log_mod.update_state(
            tick=tick,
            scenario=modes,
            targets={
                "cpu_pct": targets.cpu_target_pct,
                "mem_bytes": targets.mem_target_bytes,
                "gpu_util_pct": targets.gpu_util_target_pct,
                "gpu_mem_mb": targets.gpu_mem_target_mb,
                "log_lps": targets.log_rate_lps,
                "db_qps": targets.db_qps_target,
            },
        )

    @staticmethod
    def _apply_gpu(gpu_mod: GPUModule, targets: Targets) -> None:
        gpu_mod.set_targets(util_pct=targets.gpu_util_target_pct, mem_mb=targets.gpu_mem_target_mb)
        gpu_mod.poll()

    @staticmethod
    def _apply_db(db_mod: DBModule, targets: Targets, tick: int) -> None:
        db_mod.set_target_qps(targets.db_qps_target)
        db_mod.on_tick(tick)

    @staticmethod
    def _snapshot_db(db_mod: DBModule) -> dict[str, object]:
        snap = db_mod.snapshot()
        return {
            "db_ops_last_interval": snap.ops,
            "db_errors_last_interval": snap.errors,
            "db_p95_ms": snap.p95_ms,
        }

