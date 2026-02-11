from __future__ import annotations

from pathlib import Path


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def get_memory_limit_bytes() -> int | None:
    """
    Returns:
        int: cgroup memory limit (bytes)
        None: unlimited / unknown
    """

    # cgroup v2
    v2 = _read_text(Path("/sys/fs/cgroup/memory.max"))
    if v2 is not None:
        if v2 == "max":
            return None
        try:
            value = int(v2)
        except ValueError:
            return None
        if value <= 0:
            return None
        if value >= (1 << 60):
            return None
        return value

    # cgroup v1
    v1 = _read_text(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if v1 is not None:
        try:
            value = int(v1)
        except ValueError:
            return None
        if value <= 0:
            return None
        if value >= (1 << 60):
            return None
        return value

    return None


def get_rss_bytes_self() -> int | None:
    try:
        statm = Path("/proc/self/statm").read_text(encoding="utf-8").strip().split()
        if len(statm) < 2:
            return None
        resident_pages = int(statm[1])
        page_size = 4096
        return resident_pages * page_size
    except Exception:
        return None


def get_cpu_usage_usec() -> int | None:
    """
    Returns:
        int: total CPU time used by this cgroup in microseconds (cgroup v2)
        None: not available
    """

    # cgroup v2
    try:
        text = Path("/sys/fs/cgroup/cpu.stat").read_text(encoding="utf-8")
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == "usage_usec":
                return int(parts[1])
    except Exception:
        pass

    # cgroup v1 (cpuacct.usage is nanoseconds)
    try:
        ns = int(Path("/sys/fs/cgroup/cpuacct/cpuacct.usage").read_text(encoding="utf-8").strip())
        return int(ns / 1000)
    except Exception:
        return None
