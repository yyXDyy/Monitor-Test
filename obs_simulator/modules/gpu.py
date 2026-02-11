from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable


_RESTART_EXIT_CODE = 42


def _burn_gpu(torch, a, b, deadline: float) -> None:  # pragma: no cover
    # Keep kernels small-ish to avoid massive allocations while still showing util.
    while time.perf_counter() < deadline:
        _ = a @ b


def _gpu_worker(
    device_label: str,
    cmd: Connection,
    util_duty_cycle_ms: int,
    mem_free_policy: str,
    initial_util_pct: int,
    initial_mem_mb: int,
) -> None:  # pragma: no cover
    # Isolate per device by limiting visible devices in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = device_label if device_label != "all" else os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # Late import so that GPU_ENABLE=false path has zero dependency.
    import torch  # type: ignore

    if not torch.cuda.is_available():
        return

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    util_pct = int(initial_util_pct)
    mem_mb = int(initial_mem_mb)
    allocated_mb = 0
    tensors: list[Any] = []

    # Pre-allocate compute tensors for utilization loop.
    a = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    b = torch.randn((1024, 1024), device=device, dtype=torch.float16)

    duty_cycle_s = max(0.01, float(util_duty_cycle_ms) / 1000.0)

    def _ensure_mem(target_mb: int) -> None:
        nonlocal allocated_mb, tensors
        target_mb = max(0, int(target_mb))
        if target_mb < allocated_mb:
            if mem_free_policy == "none":
                return
            if mem_free_policy == "empty_cache":
                # Drop references and ask CUDA allocator to release cached blocks.
                tensors = []
                allocated_mb = 0
                torch.cuda.empty_cache()
                return
            if mem_free_policy == "subprocess-restart":
                # Most reliable "drop" is to restart the subprocess.
                raise SystemExit(_RESTART_EXIT_CODE)

        # Allocate additional chunks to reach target.
        to_add = target_mb - allocated_mb
        chunk_mb = 256
        while to_add > 0:
            mb = min(chunk_mb, to_add)
            bytes_ = mb * 1024 * 1024
            numel = max(1, bytes_ // 2)  # float16
            tensors.append(torch.empty((numel,), device=device, dtype=torch.float16))
            allocated_mb += mb
            to_add -= mb

    while True:
        while cmd.poll():
            msg = cmd.recv()
            if msg == "stop":
                return
            if isinstance(msg, dict):
                util_pct = int(msg.get("util_pct", util_pct))
                mem_mb = int(msg.get("mem_mb", mem_mb))

        util_pct = max(0, min(100, util_pct))
        try:
            _ensure_mem(mem_mb)
        except SystemExit:
            raise
        except Exception:
            # Keep running even if allocation fails.
            pass

        busy_s = duty_cycle_s * (util_pct / 100.0)
        idle_s = duty_cycle_s - busy_s

        if busy_s > 0:
            _burn_gpu(torch, a, b, time.perf_counter() + busy_s)
        if idle_s > 0:
            torch.cuda.synchronize()
            time.sleep(idle_s)


@dataclass
class _GPUProc:
    device: str
    conn: Connection
    proc: mp.Process


class GPUModule:
    def __init__(
        self,
        procs: list[_GPUProc],
        *,
        util_duty_cycle_ms: int,
        mem_free_policy: str,
        event_sink: Callable[[dict[str, Any]], None],
    ) -> None:
        self._procs = procs
        self._util_duty_cycle_ms = int(util_duty_cycle_ms)
        self._mem_free_policy = str(mem_free_policy)
        self._event_sink = event_sink
        self._stopping = False
        self._last_util = 0
        self._last_mem = 0

    @classmethod
    def start(
        cls,
        *,
        gpu_device: str,
        util_duty_cycle_ms: int,
        mem_free_policy: str,
        initial_util_pct: int,
        initial_mem_mb: int,
        event_sink: Callable[[dict[str, Any]], None],
    ) -> "GPUModule | None":
        try:
            import torch  # type: ignore
        except Exception as e:
            event_sink({"event": "gpu_skipped", "reason": f"torch_import_failed: {e}"})
            return None

        if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
            event_sink({"event": "gpu_skipped", "reason": "cuda_not_available"})
            return None

        device_labels: list[str]
        if gpu_device == "all":
            n = int(torch.cuda.device_count())
            device_labels = [str(i) for i in range(n)]
        else:
            device_labels = [gpu_device]

        ctx = mp.get_context("spawn")
        procs: list[_GPUProc] = []
        for d in device_labels:
            parent, child = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_gpu_worker,
                args=(d, child, util_duty_cycle_ms, mem_free_policy, initial_util_pct, initial_mem_mb),
                name=f"gpu-worker-{d}",
            )
            p.daemon = True
            p.start()
            procs.append(_GPUProc(device=d, conn=parent, proc=p))
        mod = cls(
            procs=procs,
            util_duty_cycle_ms=util_duty_cycle_ms,
            mem_free_policy=mem_free_policy,
            event_sink=event_sink,
        )
        mod._last_util = int(initial_util_pct)
        mod._last_mem = int(initial_mem_mb)
        return mod

    def set_targets(self, *, util_pct: int, mem_mb: int) -> None:
        self._last_util = int(util_pct)
        self._last_mem = int(mem_mb)
        msg = {"util_pct": int(util_pct), "mem_mb": int(mem_mb)}
        for w in self._procs:
            try:
                w.conn.send(msg)
            except Exception:
                continue

    def poll(self) -> None:
        if self._stopping:
            return
        # Restart workers that exited requesting restart.
        for i, w in enumerate(list(self._procs)):
            if w.proc.is_alive():
                continue
            code = w.proc.exitcode
            if code == _RESTART_EXIT_CODE:
                self._event_sink({"event": "gpu_worker_restart", "device": w.device})
                self._restart_one(i, w.device)

    def _restart_one(self, idx: int, device: str) -> None:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=True)
        p = ctx.Process(
            target=_gpu_worker,
            args=(
                device,
                child,
                self._util_duty_cycle_ms,
                self._mem_free_policy,
                self._last_util,
                self._last_mem,
            ),
            name=f"gpu-worker-{device}",
        )
        p.daemon = True
        p.start()
        self._procs[idx] = _GPUProc(device=device, conn=parent, proc=p)

    def close(self, timeout_s: float = 5.0) -> None:
        self._stopping = True
        for w in self._procs:
            try:
                w.conn.send("stop")
            except Exception:
                continue
        deadline = time.monotonic() + timeout_s
        for w in self._procs:
            remaining = max(0.0, deadline - time.monotonic())
            w.proc.join(timeout=remaining)
        for w in self._procs:
            if w.proc.is_alive():
                w.proc.terminate()
