"""Textbook memory: PDF -> page-aware chunks -> embeddings -> ChromaDB search.

Embeddings use Chroma's built-in ONNX version of ``all-MiniLM-L6-v2``. It is the
same model as sentence-transformers but needs no PyTorch, which keeps installs
small enough for Streamlit Community Cloud.

Each PDF gets its own collection named after a hash of its bytes, so re-uploading
the same file is instant and different users never search each other's books.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int


@dataclass(frozen=True)
class Hit:
    text: str
    page: int
    score: float  # cosine similarity in [-1, 1]


@dataclass(frozen=True)
class IndexResult:
    collection: str
    filename: str
    pages: int
    chunks: int
    cached: bool


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return ``(page_number, text)`` for every page that has extractable text."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def chunk_pages(pages: Sequence[tuple[int, str]], chunk_words: int = 220, overlap: int = 40) -> list[Chunk]:
    """Split each page into overlapping word windows, keeping the page number.

    Overlap stops an idea from being cut in half at a chunk boundary.
    """
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    step = chunk_words - overlap
    chunks: list[Chunk] = []
    for page, text in pages:
        words = text.split()
        for start in range(0, max(len(words) - overlap, 1), step):
            window = words[start : start + chunk_words]
            if window:
                chunks.append(Chunk(" ".join(window), page, len(chunks)))
    return chunks


def format_context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"[p. {h.page}] {h.text}" for h in hits)


EmbeddingFunction = Callable[[list[str]], Any]


class TextbookIndex:
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        embedding_function: EmbeddingFunction | None = None,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        settings = Settings(anonymized_telemetry=False)
        if persist_dir is None:
            self._client = chromadb.EphemeralClient(settings=settings)
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
        if embedding_function is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            embedding_function = DefaultEmbeddingFunction()
        self._embed = embedding_function

    def _collection(self, name: str, filename: str = ""):
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embed,
            metadata={"hnsw:space": "cosine", "filename": filename or "unknown"},
        )

    def index_pdf(self, pdf_bytes: bytes, filename: str, batch_size: int = 64) -> IndexResult:
        name = f"pdf_{fingerprint(pdf_bytes)}"
        collection = self._collection(name, filename)
        pages = extract_pages(pdf_bytes)
        if collection.count() > 0:
            return IndexResult(name, filename, len(pages), collection.count(), cached=True)
        if not pages:
            raise ValueError("No text found in this PDF. It may be a scanned image; try a text-based PDF.")

        chunks = chunk_pages(pages)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[f"{name}-{c.index}" for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{"page": c.page} for c in batch],
            )
        return IndexResult(name, filename, len(pages), len(chunks), cached=False)

    def search(self, collection_name: str, query: str, k: int = 4) -> list[Hit]:
        collection = self._collection(collection_name)
        total = collection.count()
        if total == 0:
            return []
        result = collection.query(query_texts=[query], n_results=min(k, total))
        return [
            Hit(text=doc, page=int(meta.get("page", 0)), score=round(1 - float(dist), 3))
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]
