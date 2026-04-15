from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS objectives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objectives_topic ON objectives(topic_id);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    objective_id INTEGER NOT NULL REFERENCES objectives(id) ON DELETE CASCADE,
    url TEXT,
    title TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    search_query TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_objective ON sources(objective_id);
CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash);
CREATE TABLE IF NOT EXISTS thinkings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    objective_id INTEGER NOT NULL REFERENCES objectives(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    supports_source_ids TEXT NOT NULL,
    author TEXT NOT NULL,
    created_at TEXT NOT NULL,
    supersedes_id INTEGER REFERENCES thinkings(id)
);
CREATE INDEX IF NOT EXISTS idx_thinkings_objective ON thinkings(objective_id);
CREATE INDEX IF NOT EXISTS idx_thinkings_hash ON thinkings(content_hash);
CREATE TABLE IF NOT EXISTS embeddings (
    object_kind TEXT NOT NULL,
    object_id INTEGER NOT NULL,
    terms TEXT NOT NULL,
    PRIMARY KEY (object_kind, object_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _row(row) -> dict | None:
    return dict(row) if row else None


def _decode_thinking(row) -> dict | None:
    if not row:
        return None
    d = dict(row)
    d["supports_source_ids"] = json.loads(d["supports_source_ids"])
    return d


def create_topic(conn, name: str) -> int:
    row = conn.execute("SELECT id FROM topics WHERE name = ?", (name,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute("INSERT INTO topics(name, created_at) VALUES (?, ?)", (name, _now()))
    conn.commit()
    return int(cur.lastrowid)


def get_topic(conn, topic_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone())


def get_topic_by_name(conn, name: str) -> dict | None:
    return _row(conn.execute("SELECT * FROM topics WHERE name = ?", (name,)).fetchone())


def list_topics(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM topics ORDER BY id").fetchall()]


def delete_topic(conn, topic_id: int) -> None:
    for o in conn.execute("SELECT id FROM objectives WHERE topic_id = ?", (topic_id,)).fetchall():
        delete_objective(conn, int(o["id"]))
    conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
    conn.commit()


def create_objective(conn, topic_id: int, question: str) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO objectives(topic_id, question, status, created_at, updated_at) VALUES (?,?,?,?,?)",
        (topic_id, question, "open", now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_objective(conn, objective_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM objectives WHERE id = ?", (objective_id,)).fetchone())


_OBJECTIVE_MUTABLE_COLUMNS = {"question", "status"}


def update_objective(conn, objective_id: int, **fields: Any) -> None:
    if not fields:
        return
    bad = set(fields) - _OBJECTIVE_MUTABLE_COLUMNS
    if bad:
        raise ValueError(
            f"cannot update objective columns {sorted(bad)}; "
            f"allowed: {sorted(_OBJECTIVE_MUTABLE_COLUMNS)}"
        )
    fields["updated_at"] = _now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE objectives SET {cols} WHERE id = ?", (*fields.values(), objective_id))
    conn.commit()


def query_objectives(conn, topic_id=None, status=None, since=None, until=None) -> list[dict]:
    sql = "SELECT * FROM objectives WHERE 1=1"
    args: list[Any] = []
    if topic_id is not None:
        sql += " AND topic_id = ?"; args.append(topic_id)
    if status is not None:
        sql += " AND status = ?"; args.append(status)
    if since is not None:
        sql += " AND created_at >= ?"; args.append(since)
    if until is not None:
        sql += " AND created_at <= ?"; args.append(until)
    sql += " ORDER BY id"
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def delete_objective(conn, objective_id: int) -> None:
    src_ids = [int(r["id"]) for r in conn.execute(
        "SELECT id FROM sources WHERE objective_id = ?", (objective_id,)).fetchall()]
    th_ids = [int(r["id"]) for r in conn.execute(
        "SELECT id FROM thinkings WHERE objective_id = ?", (objective_id,)).fetchall()]
    for sid in src_ids:
        conn.execute("DELETE FROM embeddings WHERE object_kind='source' AND object_id=?", (sid,))
    for tid in th_ids:
        conn.execute("DELETE FROM embeddings WHERE object_kind='thinking' AND object_id=?", (tid,))
    conn.execute("DELETE FROM sources WHERE objective_id = ?", (objective_id,))
    conn.execute("DELETE FROM thinkings WHERE objective_id = ?", (objective_id,))
    conn.execute("DELETE FROM objectives WHERE id = ?", (objective_id,))
    conn.commit()


def insert_source(conn, objective_id, url, title, content, content_hash, search_query=None) -> int:
    cur = conn.execute(
        "INSERT INTO sources(objective_id, url, title, content, content_hash, fetched_at, search_query)"
        " VALUES (?,?,?,?,?,?,?)",
        (objective_id, url, title, content, content_hash, _now(), search_query),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_source(conn, source_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone())


def list_sources(conn, objective_id: int) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM sources WHERE objective_id = ? ORDER BY id", (objective_id,)).fetchall()]


def find_source_by_hash(conn, content_hash: str) -> dict | None:
    return _row(conn.execute(
        "SELECT * FROM sources WHERE content_hash = ? LIMIT 1", (content_hash,)).fetchone())


def delete_source(conn, source_id: int) -> None:
    conn.execute("DELETE FROM embeddings WHERE object_kind='source' AND object_id=?", (source_id,))
    conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    conn.commit()


def insert_thinking(conn, objective_id, content, content_hash, supports_source_ids, author, supersedes_id=None) -> int:
    cur = conn.execute(
        "INSERT INTO thinkings(objective_id, content, content_hash, supports_source_ids, author, created_at, supersedes_id)"
        " VALUES (?,?,?,?,?,?,?)",
        (objective_id, content, content_hash, json.dumps(supports_source_ids), author, _now(), supersedes_id),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_thinking(conn, thinking_id: int) -> dict | None:
    return _decode_thinking(conn.execute(
        "SELECT * FROM thinkings WHERE id = ?", (thinking_id,)).fetchone())


def list_thinkings(conn, objective_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM thinkings WHERE objective_id = ? ORDER BY id", (objective_id,)).fetchall()
    return [_decode_thinking(r) for r in rows]


def find_thinking_by_hash(conn, content_hash: str) -> dict | None:
    return _decode_thinking(conn.execute(
        "SELECT * FROM thinkings WHERE content_hash = ? LIMIT 1", (content_hash,)).fetchone())


def upsert_embedding(conn, kind: str, object_id: int, terms: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embeddings(object_kind, object_id, terms) VALUES (?,?,?)",
        (kind, object_id, json.dumps(terms)),
    )
    conn.commit()


def get_embeddings(conn, kind: str, ids: Iterable[int] | None = None) -> list[dict]:
    if ids is None:
        rows = conn.execute(
            "SELECT object_id, terms FROM embeddings WHERE object_kind = ?", (kind,)).fetchall()
    else:
        ids = list(ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT object_id, terms FROM embeddings WHERE object_kind = ? AND object_id IN ({placeholders})",
            (kind, *ids),
        ).fetchall()
    return [{"object_id": int(r["object_id"]), "terms": json.loads(r["terms"])} for r in rows]
