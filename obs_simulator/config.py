from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass
from typing import Mapping
from urllib.parse import urlparse, urlunparse

from obs_simulator.utils.cgroup import get_memory_limit_bytes
from obs_simulator.utils.parse import (
    ParseError,
    clamp_int,
    duration_to_ticks,
    maybe_env,
    parse_bool,
    parse_bytes,
    parse_bytes_list,
    parse_duration_seconds,
    parse_int,
    parse_int_list,
    parse_kv_csv,
    parse_optional_bytes,
    parse_optional_bytes_list,
    parse_optional_duration_seconds,
    parse_optional_int,
    parse_optional_int_list,
    kv_list_to_dict,
)


WAVEFORM_MODES = {"steady", "step", "spike", "jitter", "sine", "random"}
LOG_MODES = {"steady", "spike", "jitter", "random"}
OBJECT_TYPES = {"app", "devbox", "db"}
GPU_MEM_FREE_POLICIES = {"none", "empty_cache", "subprocess-restart"}
DB_TYPES = {"postgres", "mysql", "redis", "mongo", "auto"}
DB_CHAOS = {"none", "slow", "lock", "conn_storm", "timeout"}
KB_TLS_MODES = {"auto", "disable", "require"}


def _get(env: Mapping[str, str], key: str) -> str | None:
    return maybe_env(env.get(key))


def _get_str(env: Mapping[str, str], key: str, default: str) -> str:
    v = _get(env, key)
    return default if v is None else v


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    v = _get(env, key)
    return default if v is None else parse_int(v)


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    v = _get(env, key)
    return default if v is None else parse_bool(v)


def _get_enum(env: Mapping[str, str], key: str, default: str, allowed: set[str]) -> str:
    v = _get(env, key)
    val = default if v is None else v.strip().lower()
    if val not in allowed:
        raise ParseError(f"{key} must be one of {sorted(allowed)}, got {v!r}")
    return val


def _get_optional_enum(env: Mapping[str, str], key: str, allowed: set[str]) -> str | None:
    v = _get(env, key)
    if v is None:
        return None
    val = v.strip().lower()
    if val not in allowed:
        raise ParseError(f"{key} must be one of {sorted(allowed)}, got {v!r}")
    return val


def redact_uri(uri: str) -> str:
    p = urlparse(uri)
    if not p.scheme:
        return uri
    if p.password is None:
        return uri
    netloc = ""
    if p.username is not None:
        netloc += p.username
    netloc += ":***@"
    if p.hostname is not None:
        netloc += p.hostname
    if p.port is not None:
        netloc += f":{p.port}"
    return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))


@dataclass(frozen=True)
class GlobalConfig:
    sim_id: str
    object_type: str
    duration_seconds: float  # 0 => infinite
    interval_seconds: float
    sim_mode: str
    seed: int
    log_level: str
    strict_idle_mode: bool
    heartbeat_enable: bool
    event_log_enable: bool


@dataclass(frozen=True)
class CPUConfig:
    enable: bool
    mode: str | None
    cores: str
    cycle_ms: int
    target_pct: int
    jitter_pct: int
    spike_pct: int
    spike_every_ticks: int
    spike_last_ticks: int
    step_series: list[int]
    step_every_ticks: int
    sine_min: int
    sine_max: int
    sine_period_ticks: int
    random_min: int
    random_max: int
    outlier_enable: bool
    outlier_every_ticks: int
    outlier_pct: int


@dataclass(frozen=True)
class MemConfig:
    enable: bool
    mode: str | None
    target_bytes: int | None
    target_pct: int
    touch: bool
    release_smooth: bool
    jitter_bytes: int
    spike_bytes: int
    spike_every_ticks: int
    spike_last_ticks: int
    step_series: list[int]
    step_every_ticks: int
    sine_min_bytes: int
    sine_max_bytes: int
    sine_period_ticks: int
    random_min_bytes: int
    random_max_bytes: int
    leak_bytes_per_sec: int
    oom_pct: int
    outlier_enable: bool
    outlier_every_ticks: int
    outlier_bytes: int


@dataclass(frozen=True)
class GPUConfig:
    enable: bool
    device: str
    mode: str | None
    mem_mode: str | None
    util_target_pct: int
    util_jitter_pct: int
    util_spike_pct: int
    util_spike_every_ticks: int
    util_spike_last_ticks: int
    util_step_series: list[int]
    util_step_every_ticks: int
    util_sine_min: int
    util_sine_max: int
    util_sine_period_ticks: int
    util_duty_cycle_ms: int
    mem_target_mb: int
    mem_jitter_mb: int
    mem_spike_mb: int
    mem_spike_every_ticks: int
    mem_spike_last_ticks: int
    mem_step_series_mb: list[int]
    mem_step_every_ticks: int
    mem_sine_min_mb: int
    mem_sine_max_mb: int
    mem_sine_period_ticks: int
    mem_free_policy: str
    outlier_enable: bool
    outlier_every_ticks: int
    outlier_util_pct: int
    outlier_mem_mb: int


@dataclass(frozen=True)
class LogConfig:
    enable: bool
    fmt: str
    mode: str
    rate: int
    jitter_pct: int
    burst_rate: int
    burst_every_ticks: int
    burst_last_ticks: int
    error_pct: int
    multiline_pct: int
    max_line_bytes: int
    fields_extra: dict[str, str]


@dataclass(frozen=True)
class KubeblocksConfig:
    enable: bool
    namespace: str | None
    cluster_name: str | None
    component: str | None
    service_role: str
    auth_secret: str | None
    db_name: str | None
    tls_mode: str


@dataclass(frozen=True)
class DBConfig:
    enable: str  # auto|true|false
    db_type: str
    uri: str | None
    mode: str
    qps: int
    workers: int
    read_pct: int
    payload_bytes: int
    keyspace: int
    init: bool
    jitter_pct: int
    spike_qps: int | None
    spike_every_ticks: int
    spike_last_ticks: int
    step_series: list[int]
    step_every_ticks: int
    sine_min_qps: int
    sine_max_qps: int
    sine_period_ticks: int
    random_min_qps: int
    random_max_qps: int
    chaos: str
    slow_ms: int
    slow_pct: int
    lock_pct: int
    timeout_ms: int
    conn_storm_conn: int
    conn_storm_every_ticks: int
    conn_storm_last_ticks: int
    kubeblocks: KubeblocksConfig


@dataclass(frozen=True)
class SimulatorConfig:
    global_cfg: GlobalConfig
    cpu: CPUConfig
    mem: MemConfig
    gpu: GPUConfig
    log: LogConfig
    db: DBConfig
    memory_limit_bytes: int | None

    def to_effective_dict(self) -> dict:
        g = self.global_cfg
        out = {
            "sim_id": g.sim_id,
            "object_type": g.object_type,
            "sim_duration_s": g.duration_seconds,
            "sim_interval_s": g.interval_seconds,
            "sim_mode": g.sim_mode,
            "sim_seed": g.seed,
            "strict_idle_mode": g.strict_idle_mode,
            "heartbeat_enable": g.heartbeat_enable,
            "event_log_enable": g.event_log_enable,
            "memory_limit_bytes": self.memory_limit_bytes,
            "cpu": asdict(self.cpu),
            "mem": asdict(self.mem),
            "gpu": asdict(self.gpu),
            "log": asdict(self.log),
            "db": asdict(self.db),
        }
        if out["db"].get("uri"):
            out["db"]["uri"] = redact_uri(str(out["db"]["uri"]))
        return out


def load_config(env: Mapping[str, str] | None = None) -> SimulatorConfig:
    env = os.environ if env is None else env

    # Global
    sim_id = _get(env, "SIM_ID") or str(uuid.uuid4())
    object_type = _get_enum(env, "OBJECT_TYPE", "app", OBJECT_TYPES)

    sim_duration_raw = _get(env, "SIM_DURATION")
    sim_duration_s = 0.0 if sim_duration_raw is None else parse_duration_seconds(sim_duration_raw)

    sim_interval_raw = _get(env, "SIM_INTERVAL") or "5s"
    sim_interval_s = parse_duration_seconds(sim_interval_raw)
    if sim_interval_s <= 0:
        raise ParseError("SIM_INTERVAL must be > 0")

    sim_mode = _get_enum(env, "SIM_MODE", "steady", WAVEFORM_MODES)
    sim_seed = _get_int(env, "SIM_SEED", 42)
    sim_log_level = _get_enum(env, "SIM_LOG_LEVEL", "info", {"debug", "info", "warn", "error"})
    strict_idle_mode = _get_bool(env, "STRICT_IDLE_MODE", False)
    heartbeat_enable = _get_bool(env, "HEARTBEAT_ENABLE", True)
    event_log_enable = _get_bool(env, "EVENT_LOG_ENABLE", True)

    global_cfg = GlobalConfig(
        sim_id=sim_id,
        object_type=object_type,
        duration_seconds=sim_duration_s,
        interval_seconds=sim_interval_s,
        sim_mode=sim_mode,
        seed=sim_seed,
        log_level=sim_log_level,
        strict_idle_mode=strict_idle_mode,
        heartbeat_enable=heartbeat_enable,
        event_log_enable=event_log_enable,
    )

    # CPU
    cpu_enable = _get_bool(env, "CPU_ENABLE", True)
    cpu_mode = _get_optional_enum(env, "CPU_MODE", WAVEFORM_MODES)
    cpu_cores = _get_str(env, "CPU_CORES", "all")
    if cpu_cores != "all":
        n = parse_int(cpu_cores)
        if n <= 0:
            raise ParseError("CPU_CORES must be 'all' or a positive int")
        cpu_cores = str(n)

    cpu_cycle_ms = _get_int(env, "CPU_CYCLE_MS", 200)
    if cpu_cycle_ms <= 0:
        raise ParseError("CPU_CYCLE_MS must be > 0")

    cpu_target_pct = clamp_int(_get_int(env, "CPU_TARGET_PCT", 0), 0, 100)
    cpu_jitter_pct = clamp_int(_get_int(env, "CPU_JITTER_PCT", 20), 0, 100)

    cpu_spike_pct = clamp_int(_get_int(env, "CPU_SPIKE_PCT", 100), 0, 100)
    cpu_spike_every_s = parse_duration_seconds(_get_str(env, "CPU_SPIKE_EVERY", "60s"))
    cpu_spike_last_s = parse_duration_seconds(_get_str(env, "CPU_SPIKE_LAST", "3s"))
    cpu_spike_every_ticks = duration_to_ticks(cpu_spike_every_s, sim_interval_s)
    cpu_spike_last_ticks = duration_to_ticks(cpu_spike_last_s, sim_interval_s)

    cpu_step_series = parse_optional_int_list(_get(env, "CPU_STEP_SERIES")) or []
    cpu_step_every_s = parse_duration_seconds(_get_str(env, "CPU_STEP_EVERY", "2m"))
    cpu_step_every_ticks = duration_to_ticks(cpu_step_every_s, sim_interval_s)

    cpu_sine_min = _get_int(env, "CPU_SINE_MIN", 10)
    cpu_sine_max = _get_int(env, "CPU_SINE_MAX", 80)
    cpu_sine_period_s = parse_duration_seconds(_get_str(env, "CPU_SINE_PERIOD", "120s"))
    cpu_sine_period_ticks = duration_to_ticks(cpu_sine_period_s, sim_interval_s)

    cpu_random_min = _get_int(env, "CPU_RANDOM_MIN", 0)
    cpu_random_max = _get_int(env, "CPU_RANDOM_MAX", 100)

    cpu_outlier_enable = _get_bool(env, "CPU_OUTLIER_ENABLE", False)
    cpu_outlier_every_s = parse_duration_seconds(_get_str(env, "CPU_OUTLIER_EVERY", "10m"))
    cpu_outlier_every_ticks = duration_to_ticks(cpu_outlier_every_s, sim_interval_s)
    cpu_outlier_pct = clamp_int(_get_int(env, "CPU_OUTLIER_PCT", 100), 0, 100)

    cpu_cfg = CPUConfig(
        enable=cpu_enable,
        mode=cpu_mode,
        cores=cpu_cores,
        cycle_ms=cpu_cycle_ms,
        target_pct=cpu_target_pct,
        jitter_pct=cpu_jitter_pct,
        spike_pct=cpu_spike_pct,
        spike_every_ticks=cpu_spike_every_ticks,
        spike_last_ticks=cpu_spike_last_ticks,
        step_series=cpu_step_series,
        step_every_ticks=cpu_step_every_ticks,
        sine_min=cpu_sine_min,
        sine_max=cpu_sine_max,
        sine_period_ticks=cpu_sine_period_ticks,
        random_min=cpu_random_min,
        random_max=cpu_random_max,
        outlier_enable=cpu_outlier_enable,
        outlier_every_ticks=cpu_outlier_every_ticks,
        outlier_pct=cpu_outlier_pct,
    )

    # Memory (RSS)
    mem_enable = _get_bool(env, "MEM_ENABLE", True)
    mem_mode = _get_optional_enum(env, "MEM_MODE", WAVEFORM_MODES | {"leak", "oom"})
    mem_target_bytes = parse_optional_bytes(_get(env, "MEM_TARGET_BYTES"))
    mem_target_pct = clamp_int(_get_int(env, "MEM_TARGET_PCT", 0), 0, 100)
    mem_touch = _get_bool(env, "MEM_TOUCH", True)
    mem_release_smooth = _get_bool(env, "MEM_RELEASE_SMOOTH", True)

    mem_jitter_bytes = parse_bytes(_get_str(env, "MEM_JITTER_BYTES", "200Mi"))
    mem_spike_bytes = parse_bytes(_get_str(env, "MEM_SPIKE_BYTES", "1Gi"))
    mem_spike_every_s = parse_duration_seconds(_get_str(env, "MEM_SPIKE_EVERY", "60s"))
    mem_spike_last_s = parse_duration_seconds(_get_str(env, "MEM_SPIKE_LAST", "3s"))
    mem_spike_every_ticks = duration_to_ticks(mem_spike_every_s, sim_interval_s)
    mem_spike_last_ticks = duration_to_ticks(mem_spike_last_s, sim_interval_s)

    mem_step_series = parse_optional_bytes_list(_get(env, "MEM_STEP_SERIES")) or []
    mem_step_every_s = parse_duration_seconds(_get_str(env, "MEM_STEP_EVERY", "2m"))
    mem_step_every_ticks = duration_to_ticks(mem_step_every_s, sim_interval_s)

    mem_sine_min_bytes = parse_bytes(_get_str(env, "MEM_SINE_MIN_BYTES", "100Mi"))
    mem_sine_max_bytes = parse_bytes(_get_str(env, "MEM_SINE_MAX_BYTES", "1Gi"))
    mem_sine_period_s = parse_duration_seconds(_get_str(env, "MEM_SINE_PERIOD", "120s"))
    mem_sine_period_ticks = duration_to_ticks(mem_sine_period_s, sim_interval_s)

    mem_random_min_bytes = parse_bytes(_get_str(env, "MEM_RANDOM_MIN_BYTES", "0"))
    mem_random_max_bytes = parse_bytes(_get_str(env, "MEM_RANDOM_MAX_BYTES", "1Gi"))

    mem_leak_rate = parse_bytes(_get_str(env, "MEM_LEAK_BYTES_PER_SEC", "0"))
    mem_oom_pct = _get_int(env, "MEM_OOM_PCT", 0)

    mem_outlier_enable = _get_bool(env, "MEM_OUTLIER_ENABLE", False)
    mem_outlier_every_s = parse_duration_seconds(_get_str(env, "MEM_OUTLIER_EVERY", "10m"))
    mem_outlier_every_ticks = duration_to_ticks(mem_outlier_every_s, sim_interval_s)
    mem_outlier_bytes = parse_bytes(_get_str(env, "MEM_OUTLIER_BYTES", "2Gi"))

    memory_limit_bytes = get_memory_limit_bytes()
    if memory_limit_bytes is None:
        if mem_target_bytes is None and mem_target_pct > 0:
            raise ParseError(
                "MEM_TARGET_PCT is set but cgroup memory limit is unlimited/unknown; please use MEM_TARGET_BYTES"
            )
        if mem_mode == "oom":
            raise ParseError("MEM_MODE=oom requires a finite cgroup memory limit")

    # If explicitly in OOM mode but no pct provided, use a sensible default (example in contract).
    if mem_mode == "oom" and mem_oom_pct <= 0:
        mem_oom_pct = 105

    mem_cfg = MemConfig(
        enable=mem_enable,
        mode=mem_mode,
        target_bytes=mem_target_bytes,
        target_pct=mem_target_pct,
        touch=mem_touch,
        release_smooth=mem_release_smooth,
        jitter_bytes=mem_jitter_bytes,
        spike_bytes=mem_spike_bytes,
        spike_every_ticks=mem_spike_every_ticks,
        spike_last_ticks=mem_spike_last_ticks,
        step_series=mem_step_series,
        step_every_ticks=mem_step_every_ticks,
        sine_min_bytes=mem_sine_min_bytes,
        sine_max_bytes=mem_sine_max_bytes,
        sine_period_ticks=mem_sine_period_ticks,
        random_min_bytes=mem_random_min_bytes,
        random_max_bytes=mem_random_max_bytes,
        leak_bytes_per_sec=mem_leak_rate,
        oom_pct=mem_oom_pct,
        outlier_enable=mem_outlier_enable,
        outlier_every_ticks=mem_outlier_every_ticks,
        outlier_bytes=mem_outlier_bytes,
    )

    # GPU
    gpu_enable = _get_bool(env, "GPU_ENABLE", False)
    gpu_device = _get_enum(env, "GPU_DEVICE", "0", {"0", "1", "all"})
    gpu_mode = _get_optional_enum(env, "GPU_MODE", WAVEFORM_MODES)
    gpu_mem_mode = _get_optional_enum(env, "GPU_MEM_MODE", WAVEFORM_MODES)

    gpu_util_target_pct = clamp_int(_get_int(env, "GPU_UTIL_TARGET_PCT", 0), 0, 100)
    gpu_util_jitter_pct = clamp_int(_get_int(env, "GPU_UTIL_JITTER_PCT", 20), 0, 100)
    gpu_util_spike_pct = clamp_int(_get_int(env, "GPU_UTIL_SPIKE_PCT", 100), 0, 100)
    gpu_util_spike_every_s = parse_duration_seconds(_get_str(env, "GPU_UTIL_SPIKE_EVERY", "60s"))
    gpu_util_spike_last_s = parse_duration_seconds(_get_str(env, "GPU_UTIL_SPIKE_LAST", "3s"))
    gpu_util_spike_every_ticks = duration_to_ticks(gpu_util_spike_every_s, sim_interval_s)
    gpu_util_spike_last_ticks = duration_to_ticks(gpu_util_spike_last_s, sim_interval_s)
    gpu_util_step_series = parse_optional_int_list(_get(env, "GPU_UTIL_STEP_SERIES")) or []
    gpu_util_step_every_s = parse_duration_seconds(_get_str(env, "GPU_UTIL_STEP_EVERY", "2m"))
    gpu_util_step_every_ticks = duration_to_ticks(gpu_util_step_every_s, sim_interval_s)
    gpu_util_sine_min = _get_int(env, "GPU_UTIL_SINE_MIN", 10)
    gpu_util_sine_max = _get_int(env, "GPU_UTIL_SINE_MAX", 80)
    gpu_util_sine_period_s = parse_duration_seconds(_get_str(env, "GPU_UTIL_SINE_PERIOD", "120s"))
    gpu_util_sine_period_ticks = duration_to_ticks(gpu_util_sine_period_s, sim_interval_s)
    gpu_util_duty_cycle_ms = _get_int(env, "GPU_UTIL_DUTY_CYCLE_MS", 200)

    gpu_mem_target_mb = _get_int(env, "GPU_MEM_TARGET_MB", 0)
    gpu_mem_jitter_mb = _get_int(env, "GPU_MEM_JITTER_MB", 500)
    gpu_mem_spike_mb = _get_int(env, "GPU_MEM_SPIKE_MB", 2000)
    gpu_mem_spike_every_s = parse_duration_seconds(_get_str(env, "GPU_MEM_SPIKE_EVERY", "60s"))
    gpu_mem_spike_last_s = parse_duration_seconds(_get_str(env, "GPU_MEM_SPIKE_LAST", "3s"))
    gpu_mem_spike_every_ticks = duration_to_ticks(gpu_mem_spike_every_s, sim_interval_s)
    gpu_mem_spike_last_ticks = duration_to_ticks(gpu_mem_spike_last_s, sim_interval_s)
    gpu_mem_step_series = parse_optional_int_list(_get(env, "GPU_MEM_STEP_SERIES_MB")) or []
    gpu_mem_step_every_s = parse_duration_seconds(_get_str(env, "GPU_MEM_STEP_EVERY", "2m"))
    gpu_mem_step_every_ticks = duration_to_ticks(gpu_mem_step_every_s, sim_interval_s)
    gpu_mem_sine_min = _get_int(env, "GPU_MEM_SINE_MIN_MB", 0)
    gpu_mem_sine_max = _get_int(env, "GPU_MEM_SINE_MAX_MB", 6000)
    gpu_mem_sine_period_s = parse_duration_seconds(_get_str(env, "GPU_MEM_SINE_PERIOD", "120s"))
    gpu_mem_sine_period_ticks = duration_to_ticks(gpu_mem_sine_period_s, sim_interval_s)

    gpu_mem_free_policy = _get_enum(env, "GPU_MEM_FREE_POLICY", "subprocess-restart", GPU_MEM_FREE_POLICIES)

    gpu_outlier_enable = _get_bool(env, "GPU_OUTLIER_ENABLE", False)
    gpu_outlier_every_s = parse_duration_seconds(_get_str(env, "GPU_OUTLIER_EVERY", "10m"))
    gpu_outlier_every_ticks = duration_to_ticks(gpu_outlier_every_s, sim_interval_s)
    gpu_outlier_util_pct = clamp_int(_get_int(env, "GPU_OUTLIER_UTIL_PCT", 100), 0, 100)
    gpu_outlier_mem_mb = _get_int(env, "GPU_OUTLIER_MEM_MB", 0)

    gpu_cfg = GPUConfig(
        enable=gpu_enable,
        device=gpu_device,
        mode=gpu_mode,
        mem_mode=gpu_mem_mode,
        util_target_pct=gpu_util_target_pct,
        util_jitter_pct=gpu_util_jitter_pct,
        util_spike_pct=gpu_util_spike_pct,
        util_spike_every_ticks=gpu_util_spike_every_ticks,
        util_spike_last_ticks=gpu_util_spike_last_ticks,
        util_step_series=gpu_util_step_series,
        util_step_every_ticks=gpu_util_step_every_ticks,
        util_sine_min=gpu_util_sine_min,
        util_sine_max=gpu_util_sine_max,
        util_sine_period_ticks=gpu_util_sine_period_ticks,
        util_duty_cycle_ms=gpu_util_duty_cycle_ms,
        mem_target_mb=gpu_mem_target_mb,
        mem_jitter_mb=gpu_mem_jitter_mb,
        mem_spike_mb=gpu_mem_spike_mb,
        mem_spike_every_ticks=gpu_mem_spike_every_ticks,
        mem_spike_last_ticks=gpu_mem_spike_last_ticks,
        mem_step_series_mb=gpu_mem_step_series,
        mem_step_every_ticks=gpu_mem_step_every_ticks,
        mem_sine_min_mb=gpu_mem_sine_min,
        mem_sine_max_mb=gpu_mem_sine_max,
        mem_sine_period_ticks=gpu_mem_sine_period_ticks,
        mem_free_policy=gpu_mem_free_policy,
        outlier_enable=gpu_outlier_enable,
        outlier_every_ticks=gpu_outlier_every_ticks,
        outlier_util_pct=gpu_outlier_util_pct,
        outlier_mem_mb=gpu_outlier_mem_mb,
    )

    # LOG
    log_enable = _get_bool(env, "LOG_ENABLE", True)
    log_format = _get_enum(env, "LOG_FORMAT", "json", {"json", "text"})
    log_mode = _get_enum(env, "LOG_MODE", "steady", LOG_MODES)
    log_rate = _get_int(env, "LOG_RATE", 10)
    log_jitter_pct = clamp_int(_get_int(env, "LOG_JITTER_PCT", 50), 0, 100)
    log_burst_rate = _get_int(env, "LOG_BURST_RATE", 2000)
    log_burst_every_s = parse_duration_seconds(_get_str(env, "LOG_BURST_EVERY", "5m"))
    log_burst_last_s = parse_duration_seconds(_get_str(env, "LOG_BURST_LAST", "10s"))
    log_burst_every_ticks = duration_to_ticks(log_burst_every_s, sim_interval_s)
    log_burst_last_ticks = duration_to_ticks(log_burst_last_s, sim_interval_s)
    log_error_pct = clamp_int(_get_int(env, "LOG_ERROR_PCT", 10), 0, 100)
    log_multiline_pct = clamp_int(_get_int(env, "LOG_MULTILINE_PCT", 0), 0, 100)
    log_max_line_bytes = _get_int(env, "LOG_MAX_LINE_BYTES", 8192)
    log_fields_extra_raw = _get(env, "LOG_FIELDS_EXTRA") or ""
    fields_extra = kv_list_to_dict(parse_kv_csv(log_fields_extra_raw)) if log_fields_extra_raw else {}

    log_cfg = LogConfig(
        enable=log_enable,
        fmt=log_format,
        mode=log_mode,
        rate=log_rate,
        jitter_pct=log_jitter_pct,
        burst_rate=log_burst_rate,
        burst_every_ticks=log_burst_every_ticks,
        burst_last_ticks=log_burst_last_ticks,
        error_pct=log_error_pct,
        multiline_pct=log_multiline_pct,
        max_line_bytes=log_max_line_bytes,
        fields_extra=fields_extra,
    )

    # Kubeblocks
    kb_enable = _get_bool(env, "KB_ENABLE", False)
    kb_namespace = _get(env, "KB_NAMESPACE")
    kb_cluster_name = _get(env, "KB_CLUSTER_NAME")
    kb_component = _get(env, "KB_COMPONENT")
    kb_service_role = _get_enum(env, "KB_SERVICE_ROLE", "primary", {"primary", "proxy", "replica"})
    kb_auth_secret = _get(env, "KB_AUTH_SECRET")
    kb_db_name = _get(env, "KB_DB_NAME")
    kb_tls = _get_enum(env, "KB_TLS", "auto", KB_TLS_MODES)

    kubeblocks_cfg = KubeblocksConfig(
        enable=kb_enable,
        namespace=kb_namespace,
        cluster_name=kb_cluster_name,
        component=kb_component,
        service_role=kb_service_role,
        auth_secret=kb_auth_secret,
        db_name=kb_db_name,
        tls_mode=kb_tls,
    )

    # DB
    db_enable_raw = _get(env, "DB_ENABLE") or "auto"
    db_enable_val = db_enable_raw.strip().lower()
    if db_enable_val not in {"auto", "true", "false", "1", "0"}:
        raise ParseError("DB_ENABLE must be auto|true|false")
    if db_enable_val == "1":
        db_enable_val = "true"
    if db_enable_val == "0":
        db_enable_val = "false"

    db_type = _get_enum(env, "DB_TYPE", "auto", DB_TYPES)
    db_uri = _get(env, "DB_URI")
    db_mode = _get_enum(env, "DB_MODE", "steady", WAVEFORM_MODES)
    db_qps = _get_int(env, "DB_QPS", 0)
    db_workers = _get_int(env, "DB_WORKERS", 20)
    db_read_pct = clamp_int(_get_int(env, "DB_READ_PCT", 80), 0, 100)
    db_payload_bytes = _get_int(env, "DB_PAYLOAD_BYTES", 1024)
    db_keyspace = _get_int(env, "DB_KEYSPACE", 100000)
    db_init = _get_bool(env, "DB_INIT", False)

    db_jitter_pct = clamp_int(_get_int(env, "DB_JITTER_PCT", 30), 0, 100)
    db_spike_qps = parse_optional_int(_get(env, "DB_SPIKE_QPS"))
    db_spike_every_s = parse_duration_seconds(_get_str(env, "DB_SPIKE_EVERY", "60s"))
    db_spike_last_s = parse_duration_seconds(_get_str(env, "DB_SPIKE_LAST", "10s"))
    db_spike_every_ticks = duration_to_ticks(db_spike_every_s, sim_interval_s)
    db_spike_last_ticks = duration_to_ticks(db_spike_last_s, sim_interval_s)
    db_step_series = parse_optional_int_list(_get(env, "DB_STEP_SERIES")) or []
    db_step_every_s = parse_duration_seconds(_get_str(env, "DB_STEP_EVERY", "2m"))
    db_step_every_ticks = duration_to_ticks(db_step_every_s, sim_interval_s)
    db_sine_min = _get_int(env, "DB_SINE_MIN_QPS", 0)
    db_sine_max = _get_int(env, "DB_SINE_MAX_QPS", 0)
    db_sine_period_s = parse_duration_seconds(_get_str(env, "DB_SINE_PERIOD", "120s"))
    db_sine_period_ticks = duration_to_ticks(db_sine_period_s, sim_interval_s)
    db_random_min = _get_int(env, "DB_RANDOM_MIN_QPS", 0)
    db_random_max = _get_int(env, "DB_RANDOM_MAX_QPS", 0)

    db_chaos = _get_enum(env, "DB_CHAOS", "none", DB_CHAOS)
    db_slow_ms = _get_int(env, "DB_SLOW_MS", 200)
    db_slow_pct = clamp_int(_get_int(env, "DB_SLOW_PCT", 10), 0, 100)
    db_lock_pct = clamp_int(_get_int(env, "DB_LOCK_PCT", 5), 0, 100)
    db_timeout_ms = _get_int(env, "DB_TIMEOUT_MS", 2000)
    db_conn_storm_conn = _get_int(env, "DB_CONN_STORM_CONN", 200)
    db_conn_storm_every_s = parse_duration_seconds(_get_str(env, "DB_CONN_STORM_EVERY", "5m"))
    db_conn_storm_last_s = parse_duration_seconds(_get_str(env, "DB_CONN_STORM_LAST", "30s"))
    db_conn_storm_every_ticks = duration_to_ticks(db_conn_storm_every_s, sim_interval_s)
    db_conn_storm_last_ticks = duration_to_ticks(db_conn_storm_last_s, sim_interval_s)

    db_cfg = DBConfig(
        enable=db_enable_val,
        db_type=db_type,
        uri=db_uri,
        mode=db_mode,
        qps=db_qps,
        workers=db_workers,
        read_pct=db_read_pct,
        payload_bytes=db_payload_bytes,
        keyspace=db_keyspace,
        init=db_init,
        jitter_pct=db_jitter_pct,
        spike_qps=db_spike_qps,
        spike_every_ticks=db_spike_every_ticks,
        spike_last_ticks=db_spike_last_ticks,
        step_series=db_step_series,
        step_every_ticks=db_step_every_ticks,
        sine_min_qps=db_sine_min,
        sine_max_qps=db_sine_max,
        sine_period_ticks=db_sine_period_ticks,
        random_min_qps=db_random_min,
        random_max_qps=db_random_max,
        chaos=db_chaos,
        slow_ms=db_slow_ms,
        slow_pct=db_slow_pct,
        lock_pct=db_lock_pct,
        timeout_ms=db_timeout_ms,
        conn_storm_conn=db_conn_storm_conn,
        conn_storm_every_ticks=db_conn_storm_every_ticks,
        conn_storm_last_ticks=db_conn_storm_last_ticks,
        kubeblocks=kubeblocks_cfg,
    )

    return SimulatorConfig(
        global_cfg=global_cfg,
        cpu=cpu_cfg,
        mem=mem_cfg,
        gpu=gpu_cfg,
        log=log_cfg,
        db=db_cfg,
        memory_limit_bytes=memory_limit_bytes,
    )
