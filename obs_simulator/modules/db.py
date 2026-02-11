from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from obs_simulator.config import DBConfig
from obs_simulator.modules.kubeblocks import KubeblocksResolveError, resolve_db_uri
from obs_simulator.utils.token_bucket import TokenBucket


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = int(round(0.95 * (len(s) - 1)))
    return float(s[k])


def _infer_db_type_from_uri(uri: str) -> str:
    scheme = urlparse(uri).scheme.lower()
    if scheme in {"postgres", "postgresql"}:
        return "postgres"
    if scheme in {"mysql"}:
        return "mysql"
    if scheme in {"redis", "rediss"}:
        return "redis"
    if scheme in {"mongo", "mongodb"}:
        return "mongo"
    return "auto"


@dataclass
class DBSnapshot:
    ops: int = 0
    errors: int = 0
    p95_ms: float | None = None


class DBModule:
    def __init__(
        self,
        *,
        cfg: DBConfig,
        sim_id: str,
        interval_seconds: float,
        event_sink,
    ) -> None:
        self._cfg = cfg
        self._sim_id = sim_id
        self._interval_s = float(interval_seconds)
        self._event_sink = event_sink

        self._stop = threading.Event()
        self._bucket = TokenBucket(rate_per_sec=0.0, capacity=1.0)
        self._threads: list[threading.Thread] = []

        self._lock = threading.Lock()
        self._ops_since_last = 0
        self._errs_since_last = 0
        self._latencies_ms: list[float] = []

        self._resolved_type = cfg.db_type
        self._resolved_uri = cfg.uri

        self._storm_on = False

    def start(self) -> None:
        if not self.is_enabled():
            return

        self._resolve_db_uri_if_needed()
        self._resolved_type = self._resolved_type if self._resolved_type != "auto" else _infer_db_type_from_uri(self._resolved_uri or "")
        if self._resolved_type == "auto":
            raise RuntimeError("DB is enabled but DB_TYPE is auto and cannot be inferred from URI")
        if not self._resolved_uri:
            raise RuntimeError("DB is enabled but DB_URI is empty and KB resolution did not provide a URI")

        if self._cfg.init:
            self._init_db()

        for i in range(max(1, int(self._cfg.workers))):
            t = threading.Thread(target=self._worker, name=f"db-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        self._bucket.set_rate(0.0)
        for t in self._threads:
            t.join(timeout=2.0)

    def is_enabled(self) -> bool:
        if self._cfg.enable == "false":
            return False
        if self._cfg.enable == "true":
            return True
        if self._cfg.uri:
            return True
        if self._cfg.kubeblocks.enable:
            return True
        return False

    def set_target_qps(self, qps: int) -> None:
        rate = max(0.0, float(qps))
        cap = max(1.0, rate) if rate > 0 else 1.0
        self._bucket.set_rate(rate, capacity=cap)

    def on_tick(self, tick: int) -> None:
        # Connection storm chaos is tick-aligned.
        if self._cfg.chaos != "conn_storm":
            return
        every = max(1, int(self._cfg.conn_storm_every_ticks))
        last = max(1, int(self._cfg.conn_storm_last_ticks))
        on = (tick % every) < last
        if on and not self._storm_on:
            self._storm_on = True
            self._event_sink({"event": "db_conn_storm_start", "tick": tick})
            threading.Thread(target=self._conn_storm_once, name="db-conn-storm", daemon=True).start()
        if (not on) and self._storm_on:
            self._storm_on = False
            self._event_sink({"event": "db_conn_storm_end", "tick": tick})

    def snapshot(self) -> DBSnapshot:
        with self._lock:
            ops = int(self._ops_since_last)
            errs = int(self._errs_since_last)
            p95_ms = _p95(self._latencies_ms)
            self._ops_since_last = 0
            self._errs_since_last = 0
            self._latencies_ms = []
            return DBSnapshot(ops=ops, errors=errs, p95_ms=p95_ms)

    def _resolve_db_uri_if_needed(self) -> None:
        if self._resolved_uri:
            return
        if not self._cfg.kubeblocks.enable:
            return
        resolved_type, uri, meta = resolve_db_uri(self._cfg.kubeblocks, self._cfg.db_type)
        self._resolved_type = resolved_type
        self._resolved_uri = uri
        self._event_sink({"event": "db_uri_resolved", "tick": 0, "meta": meta})

    def _rng(self) -> random.Random:
        # Avoid Python's per-process hash randomization.
        seed = 0
        for b in threading.current_thread().name.encode("utf-8"):
            seed = (seed * 131 + b) & 0xFFFFFFFF
        return random.Random(seed)

    def _maybe_slow(self, rng: random.Random) -> None:
        if self._cfg.chaos != "slow":
            return
        if rng.randint(1, 100) <= int(self._cfg.slow_pct):
            time.sleep(max(0.0, float(self._cfg.slow_ms) / 1000.0))

    def _worker(self) -> None:
        rng = self._rng()
        payload = b"A" * max(0, int(self._cfg.payload_bytes))
        worker_client = self._make_worker_client()
        while not self._stop.is_set():
            ok = self._bucket.acquire(1.0, stop_event=self._stop)
            if not ok:
                break

            is_read = rng.randint(1, 100) <= int(self._cfg.read_pct)
            key = rng.randint(0, max(0, int(self._cfg.keyspace) - 1)) if int(self._cfg.keyspace) > 0 else 0

            self._maybe_slow(rng)

            t0 = time.perf_counter()
            success = True
            try:
                self._do_op(worker_client, is_read=is_read, key=key, payload=payload, rng=rng)
            except Exception:
                success = False
                # Best-effort reconnect on failure.
                try:
                    self._close_worker_client(worker_client)
                except Exception:
                    pass
                worker_client = self._make_worker_client()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            with self._lock:
                self._ops_since_last += 1
                if not success:
                    self._errs_since_last += 1
                self._latencies_ms.append(dt_ms)

        try:
            self._close_worker_client(worker_client)
        except Exception:
            pass

    def _conn_storm_once(self) -> None:
        # Best-effort: open/close N extra connections/clients quickly.
        n = max(0, int(self._cfg.conn_storm_conn))
        if n <= 0:
            return
        for _ in range(n):
            if self._stop.is_set():
                return
            try:
                self._connect_and_ping()
            except Exception:
                continue

    def _connect_and_ping(self) -> None:
        if not self._resolved_uri:
            return
        if self._resolved_type == "postgres":
            conn = self._pg_connect()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            return
        if self._resolved_type == "mysql":
            conn = self._mysql_connect()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            return
        if self._resolved_type == "redis":
            r = self._redis_client()
            r.ping()
            return
        if self._resolved_type == "mongo":
            c = self._mongo_client()
            c.admin.command("ping")
            c.close()
            return

    def _init_db(self) -> None:
        if self._resolved_type in {"postgres", "mysql"}:
            self._init_sql_kv()
            return
        if self._resolved_type == "redis":
            self._init_redis()
            return
        if self._resolved_type == "mongo":
            self._init_mongo()
            return

    def _init_sql_kv(self) -> None:
        keyspace = max(0, int(self._cfg.keyspace))
        payload = b"A" * max(0, int(self._cfg.payload_bytes))
        if self._resolved_type == "postgres":
            conn = self._pg_connect()
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS kv (id BIGINT PRIMARY KEY, v BYTEA, ts TIMESTAMPTZ)"
            )
            conn.commit()
            self._bulk_upsert_pg(conn, cur, keyspace, payload)
            cur.close()
            conn.close()
            return
        if self._resolved_type == "mysql":
            conn = self._mysql_connect()
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS kv (id BIGINT PRIMARY KEY, v LONGBLOB, ts TIMESTAMP)"
            )
            conn.commit()
            self._bulk_upsert_mysql(conn, cur, keyspace, payload)
            cur.close()
            conn.close()

    def _bulk_upsert_pg(self, conn, cur, keyspace: int, payload: bytes) -> None:
        if keyspace <= 0:
            return
        batch = 1000
        for start in range(0, keyspace, batch):
            rows = [(i, payload) for i in range(start, min(keyspace, start + batch))]
            cur.executemany(
                "INSERT INTO kv(id, v, ts) VALUES (%s, %s, now()) "
                "ON CONFLICT(id) DO UPDATE SET v=EXCLUDED.v, ts=now()",
                rows,
            )
            conn.commit()

    def _bulk_upsert_mysql(self, conn, cur, keyspace: int, payload: bytes) -> None:
        if keyspace <= 0:
            return
        batch = 1000
        for start in range(0, keyspace, batch):
            rows = [(i, payload) for i in range(start, min(keyspace, start + batch))]
            cur.executemany(
                "INSERT INTO kv(id, v, ts) VALUES (%s, %s, NOW()) "
                "ON DUPLICATE KEY UPDATE v=VALUES(v), ts=NOW()",
                rows,
            )
            conn.commit()

    def _init_redis(self) -> None:
        r = self._redis_client()
        keyspace = max(0, int(self._cfg.keyspace))
        payload = b"A" * max(0, int(self._cfg.payload_bytes))
        pipe = r.pipeline()
        for i in range(keyspace):
            pipe.set(f"k:{i}".encode("utf-8"), payload)
            if (i + 1) % 1000 == 0:
                pipe.execute()
        pipe.execute()

    def _init_mongo(self) -> None:
        c = self._mongo_client()
        keyspace = max(0, int(self._cfg.keyspace))
        payload = "A" * max(0, int(self._cfg.payload_bytes))
        db_name = urlparse(self._resolved_uri or "").path.lstrip("/") or "test"
        coll = c[db_name]["kv"]
        docs = []
        for i in range(keyspace):
            docs.append({"_id": i, "v": payload, "ts": time.time()})
            if len(docs) >= 1000:
                coll.insert_many(docs, ordered=False)
                docs = []
        if docs:
            coll.insert_many(docs, ordered=False)
        c.close()

    def _do_op(self, worker_client, *, is_read: bool, key: int, payload: bytes, rng: random.Random) -> None:
        if self._cfg.chaos == "lock" and self._resolved_type in {"postgres", "mysql"}:
            if rng.randint(1, 100) <= int(self._cfg.lock_pct):
                self._do_lock_op(key=key, payload=payload)
                return

        if self._resolved_type == "postgres":
            conn, cur = worker_client
            if is_read:
                cur.execute("SELECT v FROM kv WHERE id=%s", (key,))
                cur.fetchone()
            else:
                cur.execute(
                    "INSERT INTO kv(id, v, ts) VALUES (%s, %s, now()) "
                    "ON CONFLICT(id) DO UPDATE SET v=EXCLUDED.v, ts=now()",
                    (key, payload),
                )
                conn.commit()
            return

        if self._resolved_type == "mysql":
            conn, cur = worker_client
            if is_read:
                cur.execute("SELECT v FROM kv WHERE id=%s", (key,))
                cur.fetchone()
            else:
                cur.execute(
                    "INSERT INTO kv(id, v, ts) VALUES (%s, %s, NOW()) "
                    "ON DUPLICATE KEY UPDATE v=VALUES(v), ts=NOW()",
                    (key, payload),
                )
                conn.commit()
            return

        if self._resolved_type == "redis":
            r = worker_client
            k = f"k:{key}"
            if is_read:
                r.get(k)
            else:
                r.set(k, payload)
            return

        if self._resolved_type == "mongo":
            c, coll = worker_client
            if is_read:
                coll.find_one({"_id": key})
            else:
                coll.update_one({"_id": key}, {"$set": {"v": payload.decode("utf-8", "ignore"), "ts": time.time()}}, upsert=True)
            return

        raise RuntimeError(f"unsupported db type: {self._resolved_type!r}")

    def _make_worker_client(self):
        if not self._resolved_uri:
            raise RuntimeError("DB URI is not resolved")
        if self._resolved_type == "postgres":
            conn = self._pg_connect()
            return (conn, conn.cursor())
        if self._resolved_type == "mysql":
            conn = self._mysql_connect()
            return (conn, conn.cursor())
        if self._resolved_type == "redis":
            return self._redis_client()
        if self._resolved_type == "mongo":
            c = self._mongo_client()
            db_name = urlparse(self._resolved_uri).path.lstrip("/") or "test"
            return (c, c[db_name]["kv"])
        raise RuntimeError(f"unsupported db type: {self._resolved_type!r}")

    def _close_worker_client(self, worker_client) -> None:
        if self._resolved_type in {"postgres", "mysql"}:
            conn, cur = worker_client
            try:
                cur.close()
            finally:
                conn.close()
            return
        if self._resolved_type == "redis":
            return
        if self._resolved_type == "mongo":
            c, _coll = worker_client
            c.close()
            return

    def _do_lock_op(self, *, key: int, payload: bytes) -> None:
        # Best-effort "row lock" chaos: hold a lock briefly on a fixed key.
        lock_key = 0
        if self._resolved_type == "postgres":
            conn = self._pg_connect()
            cur = conn.cursor()
            cur.execute("BEGIN")
            cur.execute("SELECT v FROM kv WHERE id=%s FOR UPDATE", (lock_key,))
            time.sleep(min(1.0, max(0.0, float(self._cfg.slow_ms) / 1000.0)))
            cur.execute(
                "INSERT INTO kv(id, v, ts) VALUES (%s, %s, now()) "
                "ON CONFLICT(id) DO UPDATE SET v=EXCLUDED.v, ts=now()",
                (lock_key, payload),
            )
            cur.execute("COMMIT")
            cur.close()
            conn.close()
            return
        if self._resolved_type == "mysql":
            conn = self._mysql_connect()
            cur = conn.cursor()
            cur.execute("START TRANSACTION")
            cur.execute("SELECT v FROM kv WHERE id=%s FOR UPDATE", (lock_key,))
            time.sleep(min(1.0, max(0.0, float(self._cfg.slow_ms) / 1000.0)))
            cur.execute(
                "INSERT INTO kv(id, v, ts) VALUES (%s, %s, NOW()) "
                "ON DUPLICATE KEY UPDATE v=VALUES(v), ts=NOW()",
                (lock_key, payload),
            )
            cur.execute("COMMIT")
            cur.close()
            conn.close()
            return

    def _pg_connect(self):
        try:
            import pg8000  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pg8000 is not installed (see requirements-extra.txt)") from e

        p = urlparse(self._resolved_uri or "")
        timeout_s = max(0.1, float(self._cfg.timeout_ms) / 1000.0)
        return pg8000.connect(
            user=p.username or "",
            password=p.password or "",
            host=p.hostname or "localhost",
            port=int(p.port or 5432),
            database=p.path.lstrip("/") or None,
            timeout=timeout_s,
        )

    def _mysql_connect(self):
        try:
            import pymysql  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyMySQL is not installed (see requirements-extra.txt)") from e

        p = urlparse(self._resolved_uri or "")
        timeout_s = max(0.1, float(self._cfg.timeout_ms) / 1000.0)
        return pymysql.connect(
            host=p.hostname or "localhost",
            user=p.username or "",
            password=p.password or "",
            port=int(p.port or 3306),
            database=p.path.lstrip("/") or None,
            connect_timeout=timeout_s,
            read_timeout=timeout_s,
            write_timeout=timeout_s,
            autocommit=False,
        )

    def _redis_client(self):
        try:
            import redis  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("redis-py is not installed (see requirements-extra.txt)") from e

        timeout_s = max(0.1, float(self._cfg.timeout_ms) / 1000.0)
        return redis.Redis.from_url(
            self._resolved_uri or "",
            socket_timeout=timeout_s,
            socket_connect_timeout=timeout_s,
            decode_responses=False,
        )

    def _mongo_client(self):
        try:
            import pymongo  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pymongo is not installed (see requirements-extra.txt)") from e

        timeout_ms = max(1, int(self._cfg.timeout_ms))
        return pymongo.MongoClient(
            self._resolved_uri or "",
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=timeout_ms,
        )
