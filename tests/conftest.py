import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeCompletions:
    """Stands in for ``client.chat.completions``; records every call."""

    def __init__(self, reply="Fake answer", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    def create(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


@pytest.fixture
def fake_client():
    def make(reply="Fake answer", error=None):
        completions = FakeCompletions(reply, error)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions

    return make


@pytest.fixture
def hash_embedding():
    """Deterministic offline embedding (bag of hashed words) so RAG tests need no download."""
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class HashEmbedding(EmbeddingFunction[Documents]):
        def __init__(self) -> None:
            pass

        def __call__(self, input: Documents) -> Embeddings:
            vectors = []
            for text in input:
                v = np.zeros(128, dtype=np.float32)
                for word in text.lower().split():
                    v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1
                vectors.append(v / (np.linalg.norm(v) or 1.0))
            return vectors

        @staticmethod
        def name() -> str:
            return "hash-test"

        def get_config(self) -> dict:
            return {}

        @staticmethod
        def build_from_config(config):
            return HashEmbedding()

    return HashEmbedding()


@pytest.fixture
def make_pdf():
    from fpdf import FPDF

    def build(pages: list[str]) -> bytes:
        pdf = FPDF()
        for text in pages:
            pdf.add_page()
            pdf.set_font("helvetica", size=11)
            pdf.multi_cell(0, 6, text)
        return bytes(pdf.output())

    return build
