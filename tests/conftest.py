import hashlib
import math
import os

import psycopg
import pytest

ADMIN_URL = os.environ.get("TEST_ADMIN_URL", "postgresql://rag:rag@localhost:5433/rag")
TEST_DB = "rag_test"


class FakeEmbeddings:
    """Embeddings déterministes sans API : sac de mots haché dans 64 dimensions."""

    dim = 64

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def embedder():
    return FakeEmbeddings()


@pytest.fixture
def conn():
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    url = ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"
    with psycopg.connect(url) as c:
        yield c
