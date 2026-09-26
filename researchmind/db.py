"""SQLite storage: papers, chunks (+ embeddings as float32 blobs), FTS5 index,
notes/highlights, collections, chat history and saved reviews."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime

import numpy as np

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT DEFAULT '',
    year INTEGER,
    doi TEXT DEFAULT '',
    abstract TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    tags TEXT DEFAULT '',
    file_name TEXT DEFAULT '',
    file_path TEXT DEFAULT '',
    num_pages INTEGER DEFAULT 0,
    status TEXT DEFAULT 'queued',
    error TEXT DEFAULT '',
    starred INTEGER DEFAULT 0,
    read_status TEXT DEFAULT 'unread',
    collection TEXT DEFAULT '',
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    idx INTEGER,
    page INTEGER,
    section TEXT DEFAULT '',
    content TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, paper_id UNINDEXED, chunk_id UNINDEXED);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT REFERENCES papers(id) ON DELETE CASCADE,
    page INTEGER,
    quote TEXT DEFAULT '',
    note TEXT DEFAULT '',
    color TEXT DEFAULT 'yellow',
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT,
    role TEXT,
    content TEXT,
    meta TEXT DEFAULT '{}',
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    title TEXT,
    paper_ids TEXT,
    sections TEXT,
    created_at TEXT
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


# ---------------- papers ----------------
def add_paper(title: str, file_name: str, file_path: str, **meta) -> str:
    pid = uuid.uuid4().hex[:12]
    with connect() as c:
        c.execute(
            "INSERT INTO papers (id,title,file_name,file_path,created_at,authors,year,doi,collection,status)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (pid, title, file_name, file_path, _now(), meta.get("authors", ""), meta.get("year"),
             meta.get("doi", ""), meta.get("collection", ""), "parsing"),
        )
    return pid


def update_paper(pid: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE papers SET {cols} WHERE id=?", (*fields.values(), pid))


def get_paper(pid: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
    return dict(r) if r else None


def list_papers(collection: str | None = None, query: str = "", starred: bool = False,
                read_status: str | None = None) -> list[dict]:
    sql = "SELECT p.*, (SELECT COUNT(*) FROM chunks ch WHERE ch.paper_id=p.id) AS n_chunks FROM papers p WHERE 1=1"
    args: list = []
    if collection:
        sql += " AND collection=?"
        args.append(collection)
    if query:
        sql += " AND (title LIKE ? OR authors LIKE ? OR tags LIKE ?)"
        args += [f"%{query}%"] * 3
    if starred:
        sql += " AND starred=1"
    if read_status:
        sql += " AND read_status=?"
        args.append(read_status)
    sql += " ORDER BY created_at DESC"
    with connect() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def list_collections() -> list[str]:
    with connect() as c:
        rows = c.execute("SELECT DISTINCT collection FROM papers WHERE collection<>'' ORDER BY collection").fetchall()
    return [r[0] for r in rows]


def delete_paper(pid: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM chunks_fts WHERE paper_id=?", (pid,))
        c.execute("DELETE FROM papers WHERE id=?", (pid,))


def find_paper_by_hash_name(file_name: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM papers WHERE file_name=?", (file_name,)).fetchone()
    return dict(r) if r else None


# ---------------- chunks ----------------
def add_chunks(pid: str, chunks: list[dict], embeddings: list[list[float]] | None) -> None:
    with connect() as c:
        c.execute("DELETE FROM chunks_fts WHERE paper_id=?", (pid,))
        c.execute("DELETE FROM chunks WHERE paper_id=?", (pid,))
        for i, ch in enumerate(chunks):
            emb = None
            if embeddings is not None and i < len(embeddings) and embeddings[i] is not None:
                emb = np.asarray(embeddings[i], dtype=np.float32).tobytes()
            cur = c.execute(
                "INSERT INTO chunks (paper_id,idx,page,section,content,embedding) VALUES (?,?,?,?,?,?)",
                (pid, i, ch.get("page"), ch.get("section", ""), ch["content"], emb),
            )
            c.execute("INSERT INTO chunks_fts (content,paper_id,chunk_id) VALUES (?,?,?)",
                      (ch["content"], pid, cur.lastrowid))


def get_chunks(paper_ids: list[str] | None = None, with_embeddings: bool = False) -> list[dict]:
    cols = "ch.id, ch.paper_id, ch.idx, ch.page, ch.section, ch.content, p.title, p.year, p.authors"
    if with_embeddings:
        cols += ", ch.embedding"
    sql = f"SELECT {cols} FROM chunks ch JOIN papers p ON p.id=ch.paper_id"
    args: list = []
    if paper_ids:
        sql += f" WHERE ch.paper_id IN ({','.join('?' * len(paper_ids))})"
        args = list(paper_ids)
    sql += " ORDER BY ch.paper_id, ch.idx"
    with connect() as c:
        rows = [dict(r) for r in c.execute(sql, args).fetchall()]
    if with_embeddings:
        for r in rows:
            b = r.pop("embedding")
            r["vec"] = np.frombuffer(b, dtype=np.float32) if b else None
    return rows


def fts_search(query: str, paper_ids: list[str] | None, limit: int = 50) -> list[tuple[int, float]]:
    """Return [(chunk_id, bm25_score)] - lower bm25 is better in SQLite, we negate."""
    import re

    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
    if not terms:
        return []
    fts_q = " OR ".join(f'"{t}"' for t in terms[:30])
    sql = "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts WHERE chunks_fts MATCH ?"
    args: list = [fts_q]
    if paper_ids:
        sql += f" AND paper_id IN ({','.join('?' * len(paper_ids))})"
        args += list(paper_ids)
    sql += " ORDER BY s LIMIT ?"
    args.append(limit)
    with connect() as c:
        try:
            return [(int(r[0]), -float(r[1])) for r in c.execute(sql, args).fetchall()]
        except sqlite3.OperationalError:
            return []


# ---------------- notes ----------------
def add_note(pid: str, quote: str, note: str, page: int | None = None, color: str = "yellow") -> None:
    with connect() as c:
        c.execute("INSERT INTO notes (paper_id,page,quote,note,color,created_at) VALUES (?,?,?,?,?,?)",
                  (pid, page, quote, note, color, _now()))


def list_notes(pid: str | None = None, query: str = "") -> list[dict]:
    sql = "SELECT n.*, p.title FROM notes n JOIN papers p ON p.id=n.paper_id WHERE 1=1"
    args: list = []
    if pid:
        sql += " AND n.paper_id=?"
        args.append(pid)
    if query:
        sql += " AND (n.quote LIKE ? OR n.note LIKE ?)"
        args += [f"%{query}%"] * 2
    sql += " ORDER BY n.created_at DESC"
    with connect() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def delete_note(nid: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM notes WHERE id=?", (nid,))


# ---------------- chat history ----------------
def add_chat(session: str, role: str, content: str, meta: dict | None = None) -> None:
    with connect() as c:
        c.execute("INSERT INTO chats (session,role,content,meta,created_at) VALUES (?,?,?,?,?)",
                  (session, role, content, json.dumps(meta or {}), _now()))


def get_chat(session: str) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM chats WHERE session=? ORDER BY id", (session,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["meta"] = json.loads(d.get("meta") or "{}")
        out.append(d)
    return out


def list_chat_sessions() -> list[str]:
    with connect() as c:
        rows = c.execute("SELECT session, MAX(id) m FROM chats GROUP BY session ORDER BY m DESC").fetchall()
    return [r[0] for r in rows]


def clear_chat(session: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM chats WHERE session=?", (session,))


# ---------------- reviews ----------------
def save_review(title: str, paper_ids: list[str], sections: dict, rid: str | None = None) -> str:
    rid = rid or uuid.uuid4().hex[:12]
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO reviews (id,title,paper_ids,sections,created_at) VALUES (?,?,?,?,?)",
                  (rid, title, json.dumps(paper_ids), json.dumps(sections), _now()))
    return rid


def list_reviews() -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM reviews ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["paper_ids"] = json.loads(d["paper_ids"])
        d["sections"] = json.loads(d["sections"])
        out.append(d)
    return out


def delete_review(rid: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM reviews WHERE id=?", (rid,))


def stats() -> dict:
    with connect() as c:
        return {
            "papers": c.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "chunks": c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "notes": c.execute("SELECT COUNT(*) FROM notes").fetchone()[0],
            "reviews": c.execute("SELECT COUNT(*) FROM reviews").fetchone()[0],
        }
