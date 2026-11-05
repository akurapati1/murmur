"""SQLite-backed job store. Safe to use from request handlers and worker threads."""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_JSON_FIELDS = {"options", "report", "chapters"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    input_path TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT,
    options TEXT NOT NULL DEFAULT '{}',
    report TEXT,
    chapters TEXT,
    title TEXT,
    author TEXT,
    output_path TEXT,
    output_format TEXT,
    duration_sec REAL,
    eta_sec REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for key in _JSON_FIELDS:
            if d.get(key):
                d[key] = json.loads(d[key])
        return d

    @staticmethod
    def _encode(fields: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in fields.items():
            out[k] = json.dumps(v) if k in _JSON_FIELDS and v is not None else v
        return out

    def create(self, *, job_id: str, filename: str, size_bytes: int, input_path: Path,
               options: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, filename, size_bytes, input_path, status, stage, options,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (job_id, filename, size_bytes, str(input_path), "queued", "queued",
                 json.dumps(options), now, now),
            )
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._decode(row)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode(r) for r in rows]  # type: ignore[misc]

    def list_by_status(self, statuses: Iterable[str]) -> list[dict[str, Any]]:
        statuses = list(statuses)
        marks = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE status IN ({marks}) ORDER BY created_at", statuses
            ).fetchall()
        return [self._decode(r) for r in rows]  # type: ignore[misc]

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields = self._encode(fields)
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id))

    def delete(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
