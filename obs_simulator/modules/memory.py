from __future__ import annotations

import gc
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection

from obs_simulator.utils.cgroup import get_rss_bytes_self


def _touch_pages(buf: bytearray, step: int = 4096) -> None:
    if not buf:
        return
    for i in range(0, len(buf), step):
        buf[i] = (buf[i] + 1) & 0xFF


def _mem_worker(
    cmd: Connection,
    metrics: Connection,
    touch: bool,
    release_smooth: bool,
    initial_target_bytes: int,
) -> None:
    blocks: list[bytearray] = []
    allocated = 0
    max_release_step = 128 * 1024 * 1024  # 128Mi per loop when smooth releasing
    max_alloc_step = 128 * 1024 * 1024  # allocate in chunks to reduce pause
    want = int(initial_target_bytes)

    while True:
        while cmd.poll():
            msg = cmd.recv()
            if msg == "stop":
                return
            if msg == "metrics":
                rss = get_rss_bytes_self() or 0
                metrics.send({"mem_alloc_bytes": int(allocated), "mem_worker_rss_bytes": int(rss)})
                continue
            want = int(msg)
            want = max(0, want)

        if allocated < want:
            need = want - allocated
            step = min(need, max_alloc_step)
            try:
                b = bytearray(step)
            except MemoryError:
                # Let OOM happen naturally if configured to do so.
                b = bytearray()
            if touch:
                _touch_pages(b)
            blocks.append(b)
            allocated += len(b)

        elif allocated > want:
            drop = allocated - want
            if release_smooth:
                drop = min(drop, max_release_step)
            # Drop from the end.
            while drop > 0 and blocks:
                blk = blocks[-1]
                if len(blk) <= drop:
                    drop -= len(blk)
                    allocated -= len(blk)
                    blocks.pop()
                else:
                    # Shrink block by slicing.
                    new_len = len(blk) - drop
                    blocks[-1] = blk[:new_len]
                    allocated -= drop
                    drop = 0
            gc.collect()

        time.sleep(0.2)


@dataclass
class MemoryModule:
    cmd: Connection
    metrics_conn: Connection
    proc: mp.Process
    _last_metrics: dict[str, int]

    @classmethod
    def start(
        cls,
        *,
        initial_target_bytes: int,
        touch: bool,
        release_smooth: bool,
    ) -> "MemoryModule":
        ctx = mp.get_context("fork")
        cmd_parent, cmd_child = ctx.Pipe(duplex=True)
        metrics_parent, metrics_child = ctx.Pipe(duplex=True)
        proc = ctx.Process(
            target=_mem_worker,
            args=(cmd_child, metrics_child, touch, release_smooth, int(initial_target_bytes)),
            name="mem-worker",
        )
        proc.daemon = True
        proc.start()
        return cls(
            cmd=cmd_parent,
            metrics_conn=metrics_parent,
            proc=proc,
            _last_metrics={"mem_alloc_bytes": 0, "mem_worker_rss_bytes": 0},
        )

    def set_target_bytes(self, target: int) -> None:
        try:
            self.cmd.send(int(max(0, target)))
        except Exception:
            return

    def metrics(self) -> dict[str, int]:
        try:
            self.cmd.send("metrics")
            if self.metrics_conn.poll(1.0):
                msg = self.metrics_conn.recv()
                if isinstance(msg, dict):
                    self._last_metrics.update({k: int(v) for k, v in msg.items()})
        except Exception:
            pass
        return dict(self._last_metrics)

    def close(self, timeout_s: float = 5.0) -> None:
        try:
            self.cmd.send("stop")
        except Exception:
            pass
        self.proc.join(timeout=timeout_s)
        if self.proc.is_alive():
            self.proc.terminate()
