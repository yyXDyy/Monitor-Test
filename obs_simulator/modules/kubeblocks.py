from __future__ import annotations

import base64
from dataclasses import asdict
from typing import Any

from obs_simulator.config import KubeblocksConfig


class KubeblocksResolveError(RuntimeError):
    pass


def _guess_db_type(component: str | None, fallback: str = "postgres") -> str:
    if not component:
        return fallback
    c = component.lower()
    if "postgres" in c or c.startswith("pg"):
        return "postgres"
    if "mysql" in c:
        return "mysql"
    if "redis" in c:
        return "redis"
    if "mongo" in c:
        return "mongo"
    return fallback


def _b64decode(v: str) -> str:
    return base64.b64decode(v.encode("utf-8")).decode("utf-8", errors="replace")


def resolve_db_uri(kb: KubeblocksConfig, db_type_hint: str) -> tuple[str, str, dict[str, Any]]:
    """
    Best-effort Kubeblocks resolver.

    This implementation intentionally keeps the logic simple and defensive:
    - Prefer explicit KB_AUTH_SECRET.
    - Prefer service names that match cluster/component/role patterns.
    - Fall back to substring search when needed.

    Returns:
        (resolved_db_type, resolved_uri, meta)
    """
    if not kb.enable:
        raise KubeblocksResolveError("KB_ENABLE is false")
    if not kb.namespace or not kb.cluster_name:
        raise KubeblocksResolveError("KB_NAMESPACE and KB_CLUSTER_NAME must be set when KB_ENABLE=true")

    try:
        from kubernetes import client, config  # type: ignore
    except Exception as e:  # pragma: no cover
        raise KubeblocksResolveError("kubernetes python client is not installed") from e

    # Load config (in-cluster first).
    try:  # pragma: no cover
        config.load_incluster_config()
    except Exception:
        try:
            config.load_kube_config()
        except Exception as e:
            raise KubeblocksResolveError("failed to load kube config (incluster/kubeconfig)") from e

    v1 = client.CoreV1Api()

    ns = kb.namespace
    cluster = kb.cluster_name
    component = kb.component
    role = kb.service_role

    resolved_db_type = db_type_hint if db_type_hint != "auto" else _guess_db_type(component)

    # 1) Resolve secret
    secret_name = kb.auth_secret
    if not secret_name:
        secrets = v1.list_namespaced_secret(ns).items
        candidates = [s for s in secrets if cluster in (s.metadata.name or "")]
        if not candidates:
            raise KubeblocksResolveError(f"no secret found for cluster {cluster!r} in namespace {ns!r}")
        secret_name = candidates[0].metadata.name

    sec = v1.read_namespaced_secret(secret_name, ns)
    data = sec.data or {}
    user_b64 = data.get("username") or data.get("user") or data.get("USER") or ""
    pass_b64 = data.get("password") or data.get("pass") or data.get("PASS") or ""
    username = _b64decode(user_b64) if user_b64 else ""
    password = _b64decode(pass_b64) if pass_b64 else ""
    if not username and resolved_db_type != "redis":
        # Redis often works without username; others usually need it.
        raise KubeblocksResolveError(f"secret {secret_name!r} missing username field")

    # 2) Resolve service + port
    services = v1.list_namespaced_service(ns).items
    service_name_candidates: list[str] = []
    if component:
        service_name_candidates.append(f"{cluster}-{component}-{role}")
        service_name_candidates.append(f"{cluster}-{component}")
    service_name_candidates.append(f"{cluster}-{role}")
    service_name_candidates.append(cluster)

    svc = None
    for name in service_name_candidates:
        for s in services:
            if (s.metadata.name or "") == name:
                svc = s
                break
        if svc is not None:
            break

    if svc is None:
        # Fallback: substring match.
        for s in services:
            if cluster in (s.metadata.name or ""):
                svc = s
                break

    if svc is None:
        raise KubeblocksResolveError(f"no service found for cluster {cluster!r} in namespace {ns!r}")

    ports = svc.spec.ports or []
    if not ports:
        raise KubeblocksResolveError(f"service {svc.metadata.name!r} has no ports")

    default_port_by_type = {"postgres": 5432, "mysql": 3306, "redis": 6379, "mongo": 27017}
    want_port = default_port_by_type.get(resolved_db_type)
    port = None
    if want_port is not None:
        for p in ports:
            if int(p.port) == want_port:
                port = int(p.port)
                break
    if port is None:
        port = int(ports[0].port)

    host = f"{svc.metadata.name}.{ns}.svc"

    # 3) Build URI
    db_name = kb.db_name or ""
    if resolved_db_type == "postgres":
        uri = f"postgres://{username}:{password}@{host}:{port}/{db_name}".rstrip("/")
        if kb.tls_mode == "require":
            uri += "?sslmode=require"
    elif resolved_db_type == "mysql":
        uri = f"mysql://{username}:{password}@{host}:{port}/{db_name}".rstrip("/")
        if kb.tls_mode == "require":
            uri += "?ssl=true"
    elif resolved_db_type == "redis":
        # Redis usually uses password only.
        auth = f":{password}@" if password else ""
        uri = f"redis://{auth}{host}:{port}/0"
        if kb.tls_mode == "require":
            uri = uri.replace("redis://", "rediss://", 1)
    elif resolved_db_type == "mongo":
        uri = f"mongodb://{username}:{password}@{host}:{port}/{db_name}".rstrip("/")
        if kb.tls_mode == "require":
            uri += "?tls=true"
    else:
        raise KubeblocksResolveError(f"unsupported resolved db type: {resolved_db_type!r}")

    meta = {
        "kb": asdict(kb),
        "resolved_db_type": resolved_db_type,
        "resolved_service": svc.metadata.name,
        "resolved_port": port,
        "resolved_secret": secret_name,
        "resolved_host": host,
    }
    return resolved_db_type, uri, meta

