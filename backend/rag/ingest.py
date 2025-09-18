from __future__ import annotations

import asyncio
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Tuple

import httpx

from ..config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE
from .types import DocumentChunk

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _is_url(source: str) -> bool:
    return bool(_URL_RE.match(source))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._skip_stack: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        if tag in {"script", "style", "noscript"}:
            self._skip_stack.append(tag)
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if self._skip_stack and tag == self._skip_stack[-1]:
            self._skip_stack.pop()
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_stack:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        else:
            self._parts.append(text)

    def get_text(self) -> Tuple[str, str | None]:
        body = _normalize_whitespace(" ".join(self._parts))
        title = _normalize_whitespace(" ".join(self._title_parts)) if self._title_parts else None
        return body, title


def _extract_from_html(html: str) -> Tuple[str, str | None]:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def _load_url(source: str) -> Tuple[str, dict]:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(source)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        meta: dict[str, str | float | int] = {"type": "url", "url": source}
        if "html" in content_type:
            text, title = _extract_from_html(response.text)
            if title:
                meta["title"] = title
            meta["content_type"] = "text/html"
            return text, meta
        text = response.text
        meta["content_type"] = content_type.split(";")[0] if content_type else "text/plain"
        return _normalize_whitespace(text), meta


def _load_file(source: str) -> Tuple[str, dict]:
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Källa saknas: {source}")
    meta: dict[str, str | float | int] = {"type": "file", "path": str(path)}
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional beroende
            raise RuntimeError("pypdf krävs för att läsa PDF-dokument.") from exc
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                pages.append(text)
        combined = "\n".join(pages)
        meta["content_type"] = "application/pdf"
        meta["title"] = path.stem
        return _normalize_whitespace(combined), meta
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta["content_type"] = "text/plain"
    meta["title"] = path.name
    return _normalize_whitespace(text), meta


def _chunk_text(text: str, source: str, metadata: dict) -> List[DocumentChunk]:
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    chunk_size = max(RAG_CHUNK_SIZE, 1)
    overlap = max(0, min(RAG_CHUNK_OVERLAP, chunk_size - 1))
    step = chunk_size - overlap
    chunks: list[DocumentChunk] = []
    for i in range(0, len(words), step):
        part_words = words[i : i + chunk_size]
        if not part_words:
            continue
        chunk_text = " ".join(part_words).strip()
        if not chunk_text:
            continue
        chunk_meta = {**metadata, "word_count": len(part_words)}
        chunks.append(DocumentChunk(text=chunk_text, source=source, order=len(chunks), metadata=chunk_meta))
    return chunks


def load_source(source: str) -> List[DocumentChunk]:
    loader = _load_url if _is_url(source) else _load_file
    text, meta = loader(source)
    return _chunk_text(text, source=source, metadata=meta)


async def load_sources_async(sources: Iterable[str]) -> List[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    loop = asyncio.get_running_loop()
    for src in sources:
        try:
            parts = await loop.run_in_executor(None, load_source, src)
        except Exception as exc:  # pragma: no cover - error path
            raise RuntimeError(f"Misslyckades att läsa '{src}': {exc}") from exc
        chunks.extend(parts)
    return chunks
