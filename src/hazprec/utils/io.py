"""Strict JSONL reading and durable appends with quarantined torn-tail recovery."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def _scan(path: Path):
    records, valid_end, tail = [], 0, b""
    if not path.exists():
        return records, valid_end, tail
    size = path.stat().st_size
    with path.open("rb") as f:
        for number, line in enumerate(f, 1):
            try:
                obj = json.loads(line) if line.strip() else None
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if f.tell() == size and not line.endswith(b"\n"):
                    tail = line
                    break
                raise ValueError(f"corrupt JSONL at {path}:{number}") from exc
            if obj is not None:
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL record is not an object at {path}:{number}")
                records.append(obj)
            valid_end = f.tell()
    return records, valid_end, tail


def read_jsonl(path: str | Path) -> list[dict]:
    records, _, tail = _scan(Path(path))
    if tail:
        warnings.warn(f"incomplete final record in {path}; writer will quarantine it on resume", RuntimeWarning)
    return records


def iter_jsonl(paths: Iterable[str | Path]) -> Iterator[dict]:
    for p in paths:
        yield from read_jsonl(p)


def record_key(rec: dict, fields: Sequence[str]) -> tuple:
    return tuple(rec.get(f) for f in fields)


class JsonlWriter:
    """Single-writer JSONL. Concurrent writers to one experiment are unsupported."""
    def __init__(self, path: str | Path, fsync: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _, valid_end, tail = _scan(self.path)
        if tail:
            backup = self.path.with_name(self.path.name + ".torn")
            index = 0
            while backup.exists():
                index += 1
                backup = self.path.with_name(self.path.name + f".torn.{index}")
            with backup.open("xb") as f:
                f.write(tail)
                f.flush()
                os.fsync(f.fileno())
            with self.path.open("r+b") as f:
                f.truncate(valid_end)
                f.flush()
                os.fsync(f.fileno())
            warnings.warn(f"quarantined incomplete record in {backup}", RuntimeWarning)
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("r+b") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.seek(0, os.SEEK_END)
                    f.write(b"\n")
                    f.flush()
                    os.fsync(f.fileno())
        self.fsync = fsync
        self._f = self.path.open("a")

    def write(self, rec: dict[str, Any]) -> None:
        self._f.write(json.dumps(rec, default=str, allow_nan=False) + "\n")
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def completed_keys(path: str | Path, fields: Sequence[str], where: dict | None = None) -> set[tuple]:
    keys = set()
    for rec in read_jsonl(path):
        if where and any(rec.get(k) != v for k, v in where.items()):
            continue
        key = record_key(rec, fields)
        if key in keys:
            raise ValueError(f"duplicate record key in {path}: {key}")
        keys.add(key)
    return keys
