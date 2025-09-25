from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List

from ..config import (
    RAG_ENABLED,
    RAG_MIN_SCORE,
    RAG_TOP_K,
)
from ..ai import create_embeddings, chat_reply_sv
from .ingest import load_sources_async
from .storage import add_chunks, query_embeddings, reset as reset_index
from .types import RetrievedChunk


@dataclass(slots=True)
class RAGAnswer:
    question: str
    answer: str
    contexts: List[RetrievedChunk]
    used_rag: bool

    def context_payload(self) -> str:
        return json.dumps([ctx.for_client() for ctx in self.contexts], ensure_ascii=False)

    def contexts_for_client(self) -> List[dict]:
        return [ctx.for_client() for ctx in self.contexts]


async def ingest_sources(sources: Iterable[str]) -> dict:
    if not RAG_ENABLED:
        raise RuntimeError("RAG är inaktiverat. Sätt RAG_ENABLED=1 för att aktivera.")
    chunks = await load_sources_async(sources)
    if not chunks:
        return {"chunks_added": 0, "sources": []}
    embeddings = await create_embeddings([chunk.text for chunk in chunks])
    added = add_chunks(chunks, embeddings)
    unique_sources = sorted({chunk.source for chunk in chunks})
    return {"chunks_added": added, "sources": unique_sources}


async def rag_answer(question: str) -> RAGAnswer:
    if not question.strip():
        raise ValueError("Frågan måste innehålla text.")
    contexts: List[RetrievedChunk] = []
    used_rag = False
    if RAG_ENABLED:
        query_vecs = await create_embeddings([question])
        if query_vecs:
            contexts = query_embeddings(query_vecs[0], top_k=RAG_TOP_K, min_score=RAG_MIN_SCORE)
            used_rag = bool(contexts)
    context_texts = []
    for idx, ctx in enumerate(contexts, start=1):
        title = ctx.metadata.get("title") or ctx.metadata.get("source")
        header = f"Källa {idx}: {title}"
        context_texts.append(f"{header}\n{ctx.text}")
    answer = await chat_reply_sv(question, context_sections=context_texts if context_texts else None)
    return RAGAnswer(question=question, answer=answer, contexts=contexts, used_rag=used_rag)


__all__ = ["RAGAnswer", "ingest_sources", "rag_answer", "reset_index"]
