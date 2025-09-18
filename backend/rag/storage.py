from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Iterable, List

import numpy as np

from ..config import RAG_DB_PATH
from .types import DocumentChunk, RetrievedChunk

_DB_CONN: sqlite3.Connection | None = None
_DB_LOCK = threading.Lock()


def _db_path() -> str:
    os.makedirs(RAG_DB_PATH, exist_ok=True)
    return os.path.join(RAG_DB_PATH, "rag_index.sqlite")


def _get_conn() -> sqlite3.Connection:
    global _DB_CONN
    if _DB_CONN is None:
        _DB_CONN = sqlite3.connect(_db_path(), check_same_thread=False)
        _DB_CONN.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source TEXT,
                ord INTEGER,
                metadata TEXT,
                text TEXT,
                embedding BLOB
            )
            """
        )
        _DB_CONN.commit()
    return _DB_CONN


def _to_blob(vector: List[float]) -> bytes:
    arr = np.asarray(vector, dtype=np.float32)
    return arr.tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def add_chunks(chunks: Iterable[DocumentChunk], embeddings: List[List[float]]) -> int:
    docs = list(chunks)
    if not docs:
        return 0
    if len(docs) != len(embeddings):
        raise ValueError("Antalet embeddings matchar inte antalet textbitar.")
    conn = _get_conn()
    with _DB_LOCK:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks (id, source, ord, metadata, text, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    doc.id,
                    doc.source,
                    doc.order,
                    json.dumps(doc.as_metadata(), ensure_ascii=False),
                    doc.text,
                    _to_blob(embed),
                )
                for doc, embed in zip(docs, embeddings)
            ],
        )
        conn.commit()
    return len(docs)


def _all_rows(conn: sqlite3.Connection) -> list[tuple]:
    cursor = conn.execute("SELECT source, metadata, text, embedding FROM chunks")
    return cursor.fetchall()


def query_embeddings(query_embedding: List[float], top_k: int, min_score: float) -> List[RetrievedChunk]:
    conn = _get_conn()
    with _DB_LOCK:
        rows = _all_rows(conn)
    if not rows:
        return []
    query_vec = np.asarray(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    retrieved: list[RetrievedChunk] = []
    for source, metadata_json, text, embedding_blob in rows:
        if not embedding_blob:
            continue
        vec = _from_blob(embedding_blob)
        denom = np.linalg.norm(vec) * query_norm
        if denom == 0:
            continue
        similarity = float(np.dot(vec, query_vec) / denom)
        if similarity < min_score:
            continue
        metadata = json.loads(metadata_json) if metadata_json else {"source": source}
        retrieved.append(RetrievedChunk(text=text, score=similarity, metadata=metadata))
    retrieved.sort(key=lambda item: item.score, reverse=True)
    return retrieved[:top_k]


def reset() -> None:
    conn = _get_conn()
    with _DB_LOCK:
        conn.execute("DELETE FROM chunks")
        conn.commit()
