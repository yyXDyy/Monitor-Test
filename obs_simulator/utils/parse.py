from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_DURATION_RE = re.compile(r"^(?P<num>\d+)(?P<unit>ms|s|m|h)$")
_BYTES_RE = re.compile(r"^(?P<num>\d+)(?P<unit>B|Ki|Mi|Gi)$")


class ParseError(ValueError):
    pass


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def parse_bool(raw: str) -> bool:
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ParseError(f"invalid bool: {raw!r}")


def parse_optional_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    s = raw.strip()
    if s == "":
        return None
    return parse_bool(s)


def parse_int(raw: str) -> int:
    try:
        return int(raw.strip())
    except ValueError as e:
        raise ParseError(f"invalid int: {raw!r}") from e


def parse_optional_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = raw.strip()
    if s == "":
        return None
    return parse_int(s)


def parse_duration_seconds(raw: str) -> float:
    s = raw.strip()
    if s == "0":
        return 0.0
    m = _DURATION_RE.match(s)
    if not m:
        raise ParseError(f"invalid duration: {raw!r} (expected 100ms/5s/2m/1h)")
    num = int(m.group("num"))
    unit = m.group("unit")
    if unit == "ms":
        return num / 1000.0
    if unit == "s":
        return float(num)
    if unit == "m":
        return float(num) * 60.0
    if unit == "h":
        return float(num) * 3600.0
    raise ParseError(f"invalid duration unit: {unit!r}")


def parse_optional_duration_seconds(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip()
    if s == "":
        return None
    return parse_duration_seconds(s)


def parse_bytes(raw: str) -> int:
    s = raw.strip()
    if s == "0":
        return 0
    m = _BYTES_RE.match(s)
    if not m:
        raise ParseError(f"invalid bytes: {raw!r} (expected 123B/10Ki/10Mi/2Gi)")
    num = int(m.group("num"))
    unit = m.group("unit")
    if unit == "B":
        return num
    if unit == "Ki":
        return num * 1024
    if unit == "Mi":
        return num * 1024 * 1024
    if unit == "Gi":
        return num * 1024 * 1024 * 1024
    raise ParseError(f"invalid bytes unit: {unit!r}")


def parse_optional_bytes(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = raw.strip()
    if s == "":
        return None
    return parse_bytes(s)


def parse_int_list(raw: str) -> list[int]:
    items = [x.strip() for x in raw.split(",")]
    out: list[int] = []
    for it in items:
        if it == "":
            continue
        out.append(parse_int(it))
    return out


def parse_optional_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    s = raw.strip()
    if s == "":
        return None
    return parse_int_list(s)


def parse_bytes_list(raw: str) -> list[int]:
    items = [x.strip() for x in raw.split(",")]
    out: list[int] = []
    for it in items:
        if it == "":
            continue
        out.append(parse_bytes(it))
    return out


def parse_optional_bytes_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    s = raw.strip()
    if s == "":
        return None
    return parse_bytes_list(s)


def duration_to_ticks(duration_seconds: float, interval_seconds: float) -> int:
    if interval_seconds <= 0:
        raise ParseError("SIM_INTERVAL must be > 0")
    if duration_seconds <= 0:
        raise ParseError("duration must be > 0 for tick conversion")
    ticks = int(round(duration_seconds / interval_seconds))
    return max(1, ticks)


def maybe_env(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = raw.strip()
    return None if s == "" else s


@dataclass(frozen=True)
class KeyValue:
    key: str
    value: str


def parse_kv_csv(raw: str) -> list[KeyValue]:
    items = [x.strip() for x in raw.split(",")]
    out: list[KeyValue] = []
    for it in items:
        if it == "":
            continue
        if "=" not in it:
            raise ParseError(f"invalid key=value item: {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "":
            raise ParseError(f"invalid key=value item: {it!r}")
        out.append(KeyValue(k, v))
    return out


def kv_list_to_dict(items: Iterable[KeyValue]) -> dict[str, str]:
    return {it.key: it.value for it in items}

