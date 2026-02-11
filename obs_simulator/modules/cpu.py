from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection


def _burn_cpu(deadline: float) -> None:
    x = 0.0001
    while time.perf_counter() < deadline:
        x = math.sin(x) * math.cos(x) + 0.000001


def _cpu_worker(cmd: Connection, cycle_ms: int, initial_target_pct: int) -> None:
    cycle_s = max(0.001, cycle_ms / 1000.0)
    pct = int(initial_target_pct)
    while True:
        # Drain queued updates; keep the latest.
        while cmd.poll():
            msg = cmd.recv()
            if msg == "stop":
                return
            pct = int(msg)
            pct = max(0, min(100, pct))

        busy_s = cycle_s * (pct / 100.0)
        idle_s = cycle_s - busy_s

        if busy_s > 0:
            _burn_cpu(time.perf_counter() + busy_s)
        if idle_s > 0:
            time.sleep(idle_s)


@dataclass
class CPUModule:
    conns: list[Connection]
    procs: list[mp.Process]

    @classmethod
    def start(cls, *, workers: int, cycle_ms: int, initial_target_pct: int) -> "CPUModule":
        ctx = mp.get_context("fork")
        procs: list[mp.Process] = []
        conns: list[Connection] = []
        for i in range(workers):
            parent, child = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_cpu_worker,
                args=(child, cycle_ms, int(initial_target_pct)),
                name=f"cpu-worker-{i}",
            )
            p.daemon = True
            p.start()
            procs.append(p)
            conns.append(parent)
        return cls(conns=conns, procs=procs)

    def set_target_pct(self, pct: int) -> None:
        v = int(max(0, min(100, pct)))
        for c in self.conns:
            try:
                c.send(v)
            except Exception:
                continue

    def close(self, timeout_s: float = 5.0) -> None:
        for c in self.conns:
            try:
                c.send("stop")
            except Exception:
                continue
        deadline = time.monotonic() + timeout_s
        for p in self.procs:
            remaining = max(0.0, deadline - time.monotonic())
            p.join(timeout=remaining)
        for p in self.procs:
            if p.is_alive():
                p.terminate()


def resolve_worker_count(cpu_cores: str) -> int:
    if cpu_cores == "all":
        return max(1, os.cpu_count() or 1)
    try:
        n = int(cpu_cores)
    except ValueError:
        return max(1, os.cpu_count() or 1)
    return max(1, n)
